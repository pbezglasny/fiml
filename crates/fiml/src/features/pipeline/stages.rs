//! Validates fitted vector stages and executes them using preallocated scratch.

use crate::features::transformers::validate_standard_scale;
use crate::{FeatureId, FeatureVector, VecFeatureVector};

/// Learned numeric state for a stage consuming the preceding complete active vector.
/// Training stays outside Rust; these definitions freeze inference and output order.
#[derive(Debug, Clone, PartialEq)]
pub enum FittedStage {
    /// Per-column centering and scaling, preserving input IDs.
    StandardScale {
        outputs: Vec<FeatureId>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    },
    /// Row-major principal axes, with effective whitening divisors (ones if disabled).
    Pca {
        outputs: Vec<FeatureId>,
        mean: Vec<f64>,
        components: Vec<Vec<f64>>,
        output_scale: Vec<f64>,
    },
}

impl FittedStage {
    /// Active output IDs in the order written by this stage.
    pub fn outputs(&self) -> &[FeatureId] {
        match self {
            Self::StandardScale { outputs, .. } | Self::Pca { outputs, .. } => outputs,
        }
    }

    pub(super) fn validate(&self, inputs: &[FeatureId]) -> Result<(), String> {
        let outputs = self.outputs();
        if inputs.is_empty() || outputs.is_empty() {
            return Err("input and output layouts must be nonempty".into());
        }
        for (index, id) in outputs.iter().enumerate() {
            if crate::features::is_reserved_feature_id(id) || outputs[..index].contains(id) {
                return Err(format!(
                    "output {:?} is reserved or duplicated",
                    id.as_str()
                ));
            }
        }
        let rows = match self {
            Self::StandardScale { .. } => 1,
            Self::Pca { .. } => outputs.len(),
        };
        inputs
            .len()
            .checked_mul(rows)
            .and_then(|size| size.checked_mul(size_of::<f64>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or("stage dimensions exceed addressable storage")?;
        match self {
            Self::StandardScale { mean, scale, .. } => {
                if outputs != inputs || mean.len() != inputs.len() || scale.len() != inputs.len() {
                    return Err(format!(
                        "scaler must preserve {} input IDs and have matching mean/scale lengths",
                        inputs.len()
                    ));
                }
                for ((id, &mean), &scale) in inputs.iter().zip(mean).zip(scale) {
                    validate_standard_scale(mean, scale)
                        .map_err(|reason| format!("input {:?}: {reason}", id.as_str()))?;
                }
            }
            Self::Pca {
                mean,
                components,
                output_scale,
                ..
            } => {
                if outputs.len() > inputs.len()
                    || mean.len() != inputs.len()
                    || components.len() != outputs.len()
                    || output_scale.len() != outputs.len()
                    || components.iter().any(|row| row.len() != inputs.len())
                {
                    return Err(format!(
                        "PCA requires mean[{}], components[{}][{}], output_scale[{}], and outputs <= inputs",
                        inputs.len(),
                        outputs.len(),
                        inputs.len(),
                        outputs.len()
                    ));
                }
                for (id, &value) in inputs.iter().zip(mean) {
                    if !value.is_finite() {
                        return Err(format!("input {:?}: PCA mean must be finite", id.as_str()));
                    }
                }
                for ((id, row), &scale) in outputs.iter().zip(components).zip(output_scale) {
                    if !scale.is_finite()
                        || scale <= 0.0
                        || row.iter().any(|v| !v.is_finite())
                        || !dot(mean, row).is_finite()
                    {
                        return Err(format!(
                            "output {:?}: PCA components/projected mean must be finite and output scale positive and finite",
                            id.as_str()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Contiguous fitted state compiled once for numeric-only event processing.
enum CompiledStage {
    StandardScale {
        mean: Box<[f64]>,
        inverse_scale: Box<[f64]>,
    },
    Pca {
        input_width: usize,
        components: Box<[f64]>,
        projected_mean: Box<[f64]>,
        output_scale: Box<[f64]>,
    },
}

impl From<&FittedStage> for CompiledStage {
    fn from(stage: &FittedStage) -> Self {
        match stage {
            FittedStage::StandardScale { mean, scale, .. } => Self::StandardScale {
                mean: mean.clone().into_boxed_slice(),
                inverse_scale: scale.iter().map(|scale| 1.0 / scale).collect(),
            },
            FittedStage::Pca {
                mean,
                components,
                output_scale,
                ..
            } => Self::Pca {
                input_width: mean.len(),
                components: components.iter().flatten().copied().collect(),
                projected_mean: components.iter().map(|row| dot(mean, row)).collect(),
                output_scale: output_scale.clone().into_boxed_slice(),
            },
        }
    }
}

impl CompiledStage {
    fn apply<V: FeatureVector>(&self, input: &[f64], output: &mut V) {
        match self {
            Self::StandardScale {
                mean,
                inverse_scale,
            } => {
                for (i, (&mean, &inverse)) in mean.iter().zip(inverse_scale).enumerate() {
                    output.set_value_at(i, (input[i] - mean) * inverse);
                }
            }
            Self::Pca {
                input_width,
                components,
                projected_mean,
                output_scale,
            } => {
                let input = &input[..*input_width];
                let ready = input.iter().all(|v| v.is_finite());
                for (i, row) in components.chunks_exact(*input_width).enumerate() {
                    let value = if ready {
                        (dot(input, row) - projected_mean[i]) / output_scale[i]
                    } else {
                        f64::NAN
                    };
                    output.set_value_at(i, value);
                }
            }
        }
    }
}

/// Scratch and compiled stages owned by a pipeline only when vector stages exist.
pub(super) struct StageRuntime {
    pub(super) input: VecFeatureVector,
    base_width: usize,
    scratch: VecFeatureVector,
    stages: Box<[CompiledStage]>,
}

impl StageRuntime {
    pub(super) fn new(base_width: usize, stages: &[FittedStage]) -> Option<Self> {
        if stages.is_empty() {
            return None;
        }
        let width = stages
            .iter()
            .map(|stage| stage.outputs().len())
            .fold(base_width, usize::max);
        Some(Self {
            base_width,
            input: VecFeatureVector::new(width),
            scratch: VecFeatureVector::new(if stages.len() > 1 { width } else { 0 }),
            stages: stages.iter().map(CompiledStage::from).collect(),
        })
    }

    pub(super) fn prepare_input(&mut self) {
        // Lags leave unavailable outputs untouched. Reused intermediate storage
        // must not leak the preceding event's transformed values into warm-up.
        for index in 0..self.base_width {
            self.input.set_value_at(index, f64::NAN);
        }
    }

    pub(super) fn apply<V: FeatureVector>(&mut self, output: &mut V) {
        let (last, preceding) = self.stages.split_last().expect("nonempty stage sequence");
        for stage in preceding {
            stage.apply(self.input.values(), &mut self.scratch);
            std::mem::swap(&mut self.input, &mut self.scratch);
        }
        last.apply(self.input.values(), output);
    }
}

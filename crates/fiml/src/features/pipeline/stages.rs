//! Validates scalar and fitted vector stages and executes them using preallocated scratch.

use std::borrow::Cow;

use crate::features::transformers::{self, Transformer, validate_standard_scale};
use crate::{FeatureId, FeatureVector, TransformerDefinition, VecFeatureVector};

/// Deployable scalar definitions or learned numeric state consuming the preceding active vector.
/// Training stays outside Rust; these definitions freeze inference and output order.
#[derive(Debug, Clone, PartialEq)]
pub enum FittedStage {
    /// Independent scalar outputs reading IDs from the preceding stage.
    /// Use successive stages when one scalar must consume another scalar's output.
    Scalar {
        transformations: Vec<TransformerDefinition>,
    },
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
    /// Scalar IDs are collected on this cold configuration path.
    pub fn outputs(&self) -> Cow<'_, [FeatureId]> {
        match self {
            Self::Scalar { transformations } => transformations
                .iter()
                .map(|definition| definition.output().clone())
                .collect::<Vec<_>>()
                .into(),
            Self::StandardScale { outputs, .. } | Self::Pca { outputs, .. } => {
                Cow::Borrowed(outputs)
            }
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
            Self::Scalar { .. } | Self::StandardScale { .. } => 1,
            Self::Pca { .. } => outputs.len(),
        };
        inputs
            .len()
            .checked_mul(rows)
            .and_then(|size| size.checked_mul(size_of::<f64>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or("stage dimensions exceed addressable storage")?;
        match self {
            Self::Scalar { transformations } => {
                transformers::validate(transformations, inputs)
                    .map_err(|error| error.to_string())?;
            }
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
    Scalar {
        operations: Box<[Transformer]>,
        output_width: usize,
    },
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

impl CompiledStage {
    fn new(stage: &FittedStage, inputs: &[FeatureId]) -> Self {
        match stage {
            FittedStage::Scalar { transformations } => Self::Scalar {
                operations: transformers::compile(transformations, inputs),
                output_width: transformations.len(),
            },
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

    fn apply<V: FeatureVector>(&mut self, input: &[f64], output: &mut V) {
        match self {
            Self::Scalar {
                operations,
                output_width,
            } => {
                // Lag warm-up must not expose values left in reused scratch storage.
                for index in 0..*output_width {
                    output.set_value_at(index, f64::NAN);
                }
                for operation in operations {
                    operation.apply(input, output);
                }
            }
            Self::StandardScale {
                mean,
                inverse_scale,
            } => {
                for (i, (&mean, &inverse)) in mean.iter().zip(inverse_scale.iter()).enumerate() {
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
    pub(super) fn new(base: &[TransformerDefinition], stages: &[FittedStage]) -> Option<Self> {
        if stages.is_empty() {
            return None;
        }
        let base_width = base.len();
        let mut width = base_width;
        let mut inputs: Cow<'_, [FeatureId]> = base
            .iter()
            .map(|definition| definition.output().clone())
            .collect::<Vec<_>>()
            .into();
        let stages = stages
            .iter()
            .map(|stage| {
                let compiled = CompiledStage::new(stage, &inputs);
                inputs = stage.outputs();
                width = width.max(inputs.len());
                compiled
            })
            .collect::<Box<[_]>>();
        Some(Self {
            base_width,
            input: VecFeatureVector::new(width),
            scratch: VecFeatureVector::new(if stages.len() > 1 { width } else { 0 }),
            stages,
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
        let (last, preceding) = self
            .stages
            .split_last_mut()
            .expect("nonempty stage sequence");
        for stage in preceding {
            stage.apply(self.input.values(), &mut self.scratch);
            std::mem::swap(&mut self.input, &mut self.scratch);
        }
        last.apply(self.input.values(), output);
    }
}

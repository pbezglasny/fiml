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
    /// Ordered column selection preserving retained input IDs.
    Select {
        outputs: Vec<FeatureId>,
        input_indices: Vec<usize>,
    },
    /// Per-column centering and scaling, preserving input IDs.
    StandardScale {
        outputs: Vec<FeatureId>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    },
    /// Per-column sklearn MinMaxScaler coefficients and optional output clipping.
    MinMaxScale {
        outputs: Vec<FeatureId>,
        scale: Vec<f64>,
        min: Vec<f64>,
        clip: Option<(f64, f64)>,
    },
    /// Per-column Box-Cox or Yeo-Johnson transform followed by effective scaling.
    PowerTransform {
        outputs: Vec<FeatureId>,
        method: String,
        lambdas: Vec<f64>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    },
    /// Replaces NaNs in retained columns and optionally appends fitted missing indicators.
    SimpleImpute {
        outputs: Vec<FeatureId>,
        retained_input_indices: Vec<usize>,
        replacement_values: Vec<f64>,
        indicator_input_indices: Vec<usize>,
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
            Self::Select { outputs, .. }
            | Self::StandardScale { outputs, .. }
            | Self::MinMaxScale { outputs, .. }
            | Self::PowerTransform { outputs, .. }
            | Self::SimpleImpute { outputs, .. }
            | Self::Pca { outputs, .. } => Cow::Borrowed(outputs),
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
            Self::Scalar { .. }
            | Self::Select { .. }
            | Self::StandardScale { .. }
            | Self::MinMaxScale { .. }
            | Self::PowerTransform { .. }
            | Self::SimpleImpute { .. } => 1,
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
            Self::Select { input_indices, .. } => {
                if outputs.len() != input_indices.len()
                    || !strictly_increasing_in_range(input_indices, inputs.len())
                {
                    return Err(format!(
                        "selection indices must be nonempty, unique, increasing, and below {}",
                        inputs.len()
                    ));
                }
                if outputs
                    .iter()
                    .zip(input_indices)
                    .any(|(output, &input_index)| output != &inputs[input_index])
                {
                    return Err("selected output IDs must match their input IDs".into());
                }
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
            Self::MinMaxScale {
                scale, min, clip, ..
            } => {
                if outputs != inputs || scale.len() != inputs.len() || min.len() != inputs.len() {
                    return Err(format!(
                        "MinMaxScaler must preserve {} input IDs and have matching scale/min lengths",
                        inputs.len()
                    ));
                }
                for ((id, &scale), &min) in inputs.iter().zip(scale).zip(min) {
                    if !scale.is_finite() || scale <= 0.0 || !min.is_finite() {
                        return Err(format!(
                            "input {:?}: MinMaxScaler scale must be positive and finite and min must be finite",
                            id.as_str()
                        ));
                    }
                }
                if let Some((lower, upper)) = clip
                    && (!lower.is_finite() || !upper.is_finite() || lower >= upper)
                {
                    return Err(
                        "MinMaxScaler clipping bounds must be finite and strictly increasing"
                            .into(),
                    );
                }
            }
            Self::PowerTransform {
                method,
                lambdas,
                mean,
                scale,
                ..
            } => {
                if outputs != inputs
                    || lambdas.len() != inputs.len()
                    || mean.len() != inputs.len()
                    || scale.len() != inputs.len()
                {
                    return Err(format!(
                        "PowerTransformer must preserve {} input IDs and have matching lambdas/mean/scale lengths",
                        inputs.len()
                    ));
                }
                if !matches!(method.as_str(), "yeo-johnson" | "box-cox") {
                    return Err("PowerTransformer method must be yeo-johnson or box-cox".into());
                }
                for (((id, &lambda), &mean), &scale) in
                    inputs.iter().zip(lambdas).zip(mean).zip(scale)
                {
                    if !lambda.is_finite() {
                        return Err(format!(
                            "input {:?}: PowerTransformer lambda must be finite",
                            id.as_str()
                        ));
                    }
                    validate_standard_scale(mean, scale)
                        .map_err(|reason| format!("input {:?}: {reason}", id.as_str()))?;
                }
            }
            Self::SimpleImpute {
                retained_input_indices,
                replacement_values,
                indicator_input_indices,
                ..
            } => {
                if retained_input_indices.len() != replacement_values.len()
                    || outputs.len() != retained_input_indices.len() + indicator_input_indices.len()
                {
                    return Err(
                        "SimpleImputer outputs, retained indices, replacements, and indicators have inconsistent lengths"
                            .into(),
                    );
                }
                if !strictly_increasing_in_range(retained_input_indices, inputs.len())
                    || !strictly_increasing_in_range(indicator_input_indices, inputs.len())
                {
                    return Err(format!(
                        "SimpleImputer input indices must be unique, increasing, and below {}",
                        inputs.len()
                    ));
                }
                if replacement_values.iter().any(|value| !value.is_finite()) {
                    return Err("SimpleImputer replacement values must be finite".into());
                }
                if outputs[..retained_input_indices.len()]
                    .iter()
                    .zip(retained_input_indices)
                    .any(|(output, &input_index)| output != &inputs[input_index])
                {
                    return Err(
                        "SimpleImputer retained output IDs must match their input IDs".into(),
                    );
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

fn strictly_increasing_in_range(indices: &[usize], upper_bound: usize) -> bool {
    indices.iter().all(|&index| index < upper_bound)
        && indices.windows(2).all(|pair| pair[0] < pair[1])
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
    Select {
        input_indices: Box<[usize]>,
    },
    StandardScale {
        mean: Box<[f64]>,
        inverse_scale: Box<[f64]>,
    },
    MinMaxScale {
        scale: Box<[f64]>,
        min: Box<[f64]>,
        clip: Option<(f64, f64)>,
    },
    PowerTransform {
        box_cox: bool,
        lambdas: Box<[f64]>,
        mean: Box<[f64]>,
        inverse_scale: Box<[f64]>,
    },
    SimpleImpute {
        retained_input_indices: Box<[usize]>,
        replacement_values: Box<[f64]>,
        indicator_input_indices: Box<[usize]>,
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
            FittedStage::Select { input_indices, .. } => Self::Select {
                input_indices: input_indices.clone().into_boxed_slice(),
            },
            FittedStage::StandardScale { mean, scale, .. } => Self::StandardScale {
                mean: mean.clone().into_boxed_slice(),
                inverse_scale: scale.iter().map(|scale| 1.0 / scale).collect(),
            },
            FittedStage::MinMaxScale {
                scale, min, clip, ..
            } => Self::MinMaxScale {
                scale: scale.clone().into_boxed_slice(),
                min: min.clone().into_boxed_slice(),
                clip: *clip,
            },
            FittedStage::PowerTransform {
                method,
                lambdas,
                mean,
                scale,
                ..
            } => Self::PowerTransform {
                box_cox: method == "box-cox",
                lambdas: lambdas.clone().into_boxed_slice(),
                mean: mean.clone().into_boxed_slice(),
                inverse_scale: scale.iter().map(|scale| 1.0 / scale).collect(),
            },
            FittedStage::SimpleImpute {
                retained_input_indices,
                replacement_values,
                indicator_input_indices,
                ..
            } => Self::SimpleImpute {
                retained_input_indices: retained_input_indices.clone().into_boxed_slice(),
                replacement_values: replacement_values.clone().into_boxed_slice(),
                indicator_input_indices: indicator_input_indices.clone().into_boxed_slice(),
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
            Self::Select { input_indices } => {
                for (output_index, &input_index) in input_indices.iter().enumerate() {
                    output.set_value_at(output_index, input[input_index]);
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
            Self::MinMaxScale { scale, min, clip } => {
                for (i, (&scale, &min)) in scale.iter().zip(min.iter()).enumerate() {
                    let value = input[i] * scale + min;
                    output.set_value_at(
                        i,
                        clip.map_or(value, |(lower, upper)| value.clamp(lower, upper)),
                    );
                }
            }
            Self::PowerTransform {
                box_cox,
                lambdas,
                mean,
                inverse_scale,
            } => {
                for (i, ((&lambda, &mean), &inverse_scale)) in lambdas
                    .iter()
                    .zip(mean.iter())
                    .zip(inverse_scale.iter())
                    .enumerate()
                {
                    let value = power_transform(input[i], lambda, *box_cox);
                    output.set_value_at(i, (value - mean) * inverse_scale);
                }
            }
            Self::SimpleImpute {
                retained_input_indices,
                replacement_values,
                indicator_input_indices,
            } => {
                for (output_index, (&input_index, &replacement)) in retained_input_indices
                    .iter()
                    .zip(replacement_values.iter())
                    .enumerate()
                {
                    let value = input[input_index];
                    output.set_value_at(
                        output_index,
                        if value.is_nan() { replacement } else { value },
                    );
                }
                let offset = retained_input_indices.len();
                for (indicator_index, &input_index) in indicator_input_indices.iter().enumerate() {
                    output.set_value_at(
                        offset + indicator_index,
                        if input[input_index].is_nan() {
                            1.0
                        } else {
                            0.0
                        },
                    );
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

fn power_transform(value: f64, lambda: f64, box_cox: bool) -> f64 {
    if value.is_nan() || (box_cox && value <= 0.0) {
        return f64::NAN;
    }
    if box_cox {
        if lambda.abs() < f64::EPSILON {
            value.ln()
        } else {
            (lambda * value.ln()).exp_m1() / lambda
        }
    } else if value >= 0.0 {
        if lambda.abs() < f64::EPSILON {
            value.ln_1p()
        } else {
            (lambda * value.ln_1p()).exp_m1() / lambda
        }
    } else if (lambda - 2.0).abs() > f64::EPSILON {
        -((2.0 - lambda) * (-value).ln_1p()).exp_m1() / (2.0 - lambda)
    } else {
        -(-value).ln_1p()
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

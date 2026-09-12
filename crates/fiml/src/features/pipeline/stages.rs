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
    /// Per-column fitted quantile tables and output-distribution boundary behavior.
    QuantileTransform {
        outputs: Vec<FeatureId>,
        output_distribution: String,
        quantiles: Vec<Vec<f64>>,
        references: Vec<f64>,
        all_nan_input_indices: Vec<usize>,
        bounds_threshold: f64,
        normal_clip: Option<(f64, f64)>,
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
            | Self::QuantileTransform { outputs, .. }
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
            | Self::QuantileTransform { .. }
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
            Self::QuantileTransform {
                output_distribution,
                quantiles,
                references,
                all_nan_input_indices,
                bounds_threshold,
                normal_clip,
                ..
            } => {
                if outputs != inputs
                    || quantiles.is_empty()
                    || references.len() != quantiles.len()
                    || quantiles.iter().any(|row| row.len() != inputs.len())
                {
                    return Err(format!(
                        "QuantileTransformer must preserve {} input IDs and have matching quantile/reference dimensions",
                        inputs.len()
                    ));
                }
                quantiles
                    .len()
                    .checked_mul(inputs.len())
                    .and_then(|values| values.checked_add(references.len()))
                    .and_then(|values| values.checked_mul(size_of::<f64>()))
                    .filter(|&bytes| bytes <= isize::MAX as usize)
                    .ok_or("QuantileTransformer dimensions exceed addressable storage")?;
                if quantiles.iter().flatten().any(|value| !value.is_finite())
                    || (0..inputs.len()).any(|column| {
                        quantiles
                            .windows(2)
                            .any(|rows| rows[0][column] > rows[1][column])
                    })
                {
                    return Err(
                        "QuantileTransformer quantiles must be finite and nondecreasing per column"
                            .into(),
                    );
                }
                if !strictly_increasing_in_range(all_nan_input_indices, inputs.len()) {
                    return Err(format!(
                        "QuantileTransformer all-NaN input indices must be unique, increasing, and below {}",
                        inputs.len()
                    ));
                }
                if references[0] != 0.0
                    || (references.len() > 1 && references[references.len() - 1] != 1.0)
                    || references.iter().any(|value| !(0.0..=1.0).contains(value))
                    || references.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err(
                        "QuantileTransformer references must run from 0 to 1 in strictly increasing finite probabilities"
                            .into(),
                    );
                }
                if !bounds_threshold.is_finite() || *bounds_threshold < 0.0 {
                    return Err(
                        "QuantileTransformer bounds threshold must be finite and nonnegative"
                            .into(),
                    );
                }
                match (output_distribution.as_str(), normal_clip) {
                    ("uniform", None) => {}
                    ("normal", Some((lower, upper)))
                        if lower.is_finite() && upper.is_finite() && lower < upper => {}
                    ("uniform", Some(_)) => {
                        return Err(
                            "uniform QuantileTransformer must not define normal clipping bounds"
                                .into(),
                        );
                    }
                    ("normal", _) => {
                        return Err(
                            "normal QuantileTransformer requires finite increasing clipping bounds"
                                .into(),
                        );
                    }
                    _ => {
                        return Err(
                            "QuantileTransformer output distribution must be uniform or normal"
                                .into(),
                        );
                    }
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
    QuantileTransform {
        normal_clip: Option<(f64, f64)>,
        all_nan_inputs: Box<[bool]>,
        quantile_count: usize,
        quantiles: Box<[f64]>,
        references: Box<[f64]>,
        bounds_threshold: f64,
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
            FittedStage::QuantileTransform {
                quantiles,
                references,
                all_nan_input_indices,
                bounds_threshold,
                normal_clip,
                ..
            } => Self::QuantileTransform {
                normal_clip: *normal_clip,
                all_nan_inputs: (0..inputs.len())
                    .map(|index| all_nan_input_indices.binary_search(&index).is_ok())
                    .collect(),
                quantile_count: quantiles.len(),
                quantiles: (0..inputs.len())
                    .flat_map(|column| quantiles.iter().map(move |row| row[column]))
                    .collect(),
                references: references.clone().into_boxed_slice(),
                bounds_threshold: *bounds_threshold,
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
            Self::QuantileTransform {
                normal_clip,
                all_nan_inputs,
                quantile_count,
                quantiles,
                references,
                bounds_threshold,
            } => {
                for (i, quantiles) in quantiles.chunks_exact(*quantile_count).enumerate() {
                    let value = if all_nan_inputs[i] {
                        all_nan_quantile_transform(input[i], *quantile_count, *normal_clip)
                    } else {
                        quantile_transform(
                            input[i],
                            quantiles,
                            references,
                            *bounds_threshold,
                            *normal_clip,
                        )
                    };
                    output.set_value_at(i, value);
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

fn quantile_transform(
    value: f64,
    quantiles: &[f64],
    references: &[f64],
    bounds_threshold: f64,
    normal_clip: Option<(f64, f64)>,
) -> f64 {
    if value.is_nan() {
        return value;
    }
    let lower = quantiles[0];
    let upper = quantiles[quantiles.len() - 1];
    let normal = normal_clip.is_some();
    let probability = if (normal && value - bounds_threshold < lower) || (!normal && value == lower)
    {
        0.0
    } else if (normal && value + bounds_threshold > upper) || (!normal && value == upper) {
        1.0
    } else {
        (interpolate_upper_tie(value, quantiles, references)
            + interpolate_lower_tie(value, quantiles, references))
            * 0.5
    };
    normal_clip.map_or(probability, |(lower, upper)| {
        inverse_normal(probability).clamp(lower, upper)
    })
}

fn all_nan_quantile_transform(
    value: f64,
    quantile_count: usize,
    normal_clip: Option<(f64, f64)>,
) -> f64 {
    if value.is_nan() || quantile_count > 1 {
        f64::NAN
    } else {
        normal_clip.map_or(0.0, |(lower, _)| lower)
    }
}

fn interpolate_upper_tie(value: f64, quantiles: &[f64], references: &[f64]) -> f64 {
    let upper = quantiles.partition_point(|&quantile| quantile <= value);
    interpolate_at(upper, value, quantiles, references)
}

fn interpolate_lower_tie(value: f64, quantiles: &[f64], references: &[f64]) -> f64 {
    let upper = quantiles.partition_point(|&quantile| quantile < value);
    if upper < quantiles.len() && quantiles[upper] == value {
        references[upper]
    } else {
        interpolate_at(upper, value, quantiles, references)
    }
}

fn interpolate_at(upper: usize, value: f64, quantiles: &[f64], references: &[f64]) -> f64 {
    if upper == 0 {
        references[0]
    } else if upper == quantiles.len() {
        references[references.len() - 1]
    } else {
        let lower = upper - 1;
        let slope = (references[upper] - references[lower]) / (quantiles[upper] - quantiles[lower]);
        references[lower] + slope * (value - quantiles[lower])
    }
}

// Wichura's AS 241 rational approximation; full f64 accuracy over (0, 1).
fn inverse_normal(probability: f64) -> f64 {
    const A: [f64; 8] = [
        2.509_080_928_730_122_7e3,
        3.343_057_558_358_813e4,
        6.726_577_092_700_87e4,
        4.592_195_393_154_987e4,
        1.373_169_376_550_946e4,
        1.971_590_950_306_551_4e3,
        1.331_416_678_917_843_8e2,
        3.387_132_872_796_366_5,
    ];
    const B: [f64; 8] = [
        5.226_495_278_852_855e3,
        2.872_908_573_572_194e4,
        3.930_789_580_009_271e4,
        2.121_379_430_158_659e4,
        5.394_196_021_424_751e3,
        6.871_870_074_920_579e2,
        4.231_333_070_160_091e1,
        1.0,
    ];
    const C: [f64; 8] = [
        7.745_450_142_783_414e-4,
        2.272_384_498_926_918_5e-2,
        2.417_807_251_774_506e-1,
        1.270_458_252_452_368_4,
        3.647_848_324_763_204_5,
        5.769_497_221_460_691,
        4.630_337_846_156_546,
        1.423_437_110_749_683_5,
    ];
    const D: [f64; 8] = [
        1.050_750_071_644_416_9e-9,
        5.475_938_084_995_345e-4,
        1.519_866_656_361_645_7e-2,
        1.481_039_764_274_800_8e-1,
        6.897_673_349_851e-1,
        1.676_384_830_183_803_8,
        2.053_191_626_637_759,
        1.0,
    ];
    const E: [f64; 8] = [
        2.010_334_399_292_288e-7,
        2.711_555_568_743_487_6e-5,
        1.242_660_947_388_078_4e-3,
        2.653_218_952_657_612_3e-2,
        2.965_605_718_285_049e-1,
        1.784_826_539_917_291_3,
        5.463_784_911_164_114,
        6.657_904_643_501_104,
    ];
    const F: [f64; 8] = [
        2.044_263_103_389_939_7e-15,
        1.421_511_758_316_446e-7,
        1.846_318_317_510_054_8e-5,
        7.868_691_311_456_133e-4,
        1.487_536_129_085_061_4e-2,
        1.369_298_809_227_358e-1,
        5.998_322_065_558_879e-1,
        1.0,
    ];

    if probability <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability >= 1.0 {
        return f64::INFINITY;
    }
    let centered = probability - 0.5;
    if centered.abs() <= 0.425 {
        let argument = 0.180_625 - centered * centered;
        centered * polynomial(argument, &A) / polynomial(argument, &B)
    } else {
        let tail = (-probability.min(1.0 - probability).ln()).sqrt();
        let value = if tail <= 5.0 {
            let argument = tail - 1.6;
            polynomial(argument, &C) / polynomial(argument, &D)
        } else {
            let argument = tail - 5.0;
            polynomial(argument, &E) / polynomial(argument, &F)
        };
        value.copysign(centered)
    }
}

fn polynomial(argument: f64, coefficients: &[f64; 8]) -> f64 {
    coefficients
        .iter()
        .skip(1)
        .fold(coefficients[0], |value, coefficient| {
            value * argument + coefficient
        })
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

//! Defines, validates, and executes scalar transformations for model input.
//!
//! Compilation resolves each transformation to indexes and precomputed parameters
//! so execution writes directly into caller-owned storage without allocation.

use std::collections::HashMap;

use crate::{
    FeatureId, FeatureVector, FimlError, InvalidTransformationDefinitionError, Result, WarmupPolicy,
};

mod average;
use average::AverageTransformer;
mod identity;
mod lagged;
mod standard_scale;

use identity::IdentityTransformer;
use lagged::LaggedFeature;
use standard_scale::StandardScaleTransformer;

const WINDOW_MAX_SIZE: usize = 10_000;

/// One named scalar transformation from an input layout to the next output layout.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerDefinition {
    /// EMA over finite source observations; missing observations emit NaN without advancing history.
    Ema {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// SMA over finite source observations; missing observations emit NaN without advancing history.
    Sma {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Copies one input scalar without changing its value.
    Identity {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
    },
    /// Emits the input scalar from `lag_window` accepted events earlier.
    /// The window must be in `1..=10_000`; output remains NaN until enough history exists.
    /// Definitions for the same input share one runtime history buffer.
    Lagged {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Number of accepted events to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Applies `(input - mean) / scale` to one input scalar.
    StandardScale {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Finite centering value.
        mean: f64,
        /// Positive finite divisor whose reciprocal must also be finite.
        scale: f64,
    },
}

impl TransformerDefinition {
    /// Creates a sample-based EMA transformation.
    pub fn ema(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::Ema {
            input,
            output,
            window,
            warmup_policy,
        }
    }

    /// Creates a sample-based SMA transformation.
    pub fn sma(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::Sma {
            input,
            output,
            window,
            warmup_policy,
        }
    }

    /// Creates a scalar identity transformation.
    pub fn identity(input: FeatureId, output: FeatureId) -> Self {
        Self::Identity { input, output }
    }

    /// Creates an event-lagged transformation with a positive history window.
    pub fn lagged(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self::Lagged {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a scalar standard-scaling transformation.
    pub fn standard_scale(input: FeatureId, output: FeatureId, mean: f64, scale: f64) -> Self {
        Self::StandardScale {
            input,
            output,
            mean,
            scale,
        }
    }

    pub(crate) fn input(&self) -> &FeatureId {
        match self {
            Self::Sma { input, .. }
            | Self::Ema { input, .. }
            | Self::Identity { input, .. }
            | Self::Lagged { input, .. }
            | Self::StandardScale { input, .. } => input,
        }
    }

    pub(crate) fn output(&self) -> &FeatureId {
        match self {
            Self::Sma { output, .. }
            | Self::Ema { output, .. }
            | Self::Identity { output, .. }
            | Self::Lagged { output, .. }
            | Self::StandardScale { output, .. } => output,
        }
    }

    pub(crate) fn validate(&self) -> std::result::Result<(), InvalidTransformationDefinitionError> {
        if matches!(
            self,
            Self::Sma { window: 0, .. } | Self::Ema { window: 0, .. }
        ) {
            return Err(InvalidTransformationDefinitionError::WindowZero);
        }
        if let Self::Lagged { lag_window, .. } = self {
            if *lag_window == 0 {
                return Err(InvalidTransformationDefinitionError::LagWindowZero);
            }
            if *lag_window > WINDOW_MAX_SIZE {
                return Err(InvalidTransformationDefinitionError::LagWindowTooLarge);
            }
        }
        if let Self::StandardScale { mean, scale, .. } = self {
            validate_standard_scale(*mean, *scale)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_standard_scale(
    mean: f64,
    scale: f64,
) -> std::result::Result<(), InvalidTransformationDefinitionError> {
    if !mean.is_finite() {
        return Err(InvalidTransformationDefinitionError::MeanNotFinite);
    }
    if !scale.is_finite() {
        return Err(InvalidTransformationDefinitionError::ScaleNotFinite);
    }
    if scale <= 0.0 {
        return Err(InvalidTransformationDefinitionError::ScaleNotPositive);
    }
    if !(1.0 / scale).is_finite() {
        return Err(InvalidTransformationDefinitionError::InverseScaleNotFinite);
    }
    Ok(())
}

/// Validates independent scalar outputs against the preceding active layout.
pub(crate) fn validate(definitions: &[TransformerDefinition], inputs: &[FeatureId]) -> Result<()> {
    for (index, definition) in definitions.iter().enumerate() {
        let reason = if !inputs.contains(definition.input()) {
            Some(InvalidTransformationDefinitionError::InputFeatureNotFound)
        } else if crate::features::is_reserved_feature_id(definition.output()) {
            Some(InvalidTransformationDefinitionError::ReservedOutputFeature)
        } else if definitions[..index]
            .iter()
            .any(|previous| previous.output() == definition.output())
        {
            Some(InvalidTransformationDefinitionError::DuplicateOutputFeature)
        } else {
            definition.validate().err()
        };
        let reason = reason.or_else(|| {
            average_key(definition).and_then(|key| {
                (definitions[..=index]
                    .iter()
                    .filter(|d| average_key(d) == Some(key))
                    .count()
                    > crate::features::MAX_OUTPUTS_PER_INDICATOR)
                    .then_some(InvalidTransformationDefinitionError::TooManyAverageOutputs)
            })
        });
        if let Some(reason) = reason {
            return Err(FimlError::InvalidTransformationDefinition { index, reason });
        }
    }
    Ok(())
}

fn average_key(definition: &TransformerDefinition) -> Option<(bool, &FeatureId, WarmupPolicy)> {
    match definition {
        TransformerDefinition::Sma {
            input,
            warmup_policy,
            ..
        } => Some((false, input, *warmup_policy)),
        TransformerDefinition::Ema {
            input,
            warmup_policy,
            ..
        } => Some((true, input, *warmup_policy)),
        _ => None,
    }
}

/// Compiles validated scalar definitions, sharing history between lags of one input.
pub(crate) fn compile(
    definitions: &[TransformerDefinition],
    inputs: &[FeatureId],
) -> Box<[Transformer]> {
    let mut operations = Vec::with_capacity(definitions.len());
    let mut averages = HashMap::<(bool, usize, WarmupPolicy), (Vec<usize>, Vec<usize>)>::new();
    let mut lagged_outputs = HashMap::<usize, (Vec<usize>, Vec<usize>)>::new();
    for (output_index, definition) in definitions.iter().enumerate() {
        let input_index = inputs
            .iter()
            .position(|id| id == definition.input())
            .expect("pipeline construction validated every input ID");
        match definition {
            TransformerDefinition::Sma { window, .. }
            | TransformerDefinition::Ema { window, .. } => {
                let (ema, _, warmup) = average_key(definition).unwrap();
                let (windows, outputs) = averages.entry((ema, input_index, warmup)).or_default();
                windows.push(*window);
                outputs.push(output_index);
            }
            TransformerDefinition::Identity { .. } => operations.push(Transformer::Identity(
                IdentityTransformer::new(input_index, output_index),
            )),
            TransformerDefinition::Lagged { lag_window, .. } => {
                let (windows, output_indices) = lagged_outputs.entry(input_index).or_default();
                windows.push(*lag_window);
                output_indices.push(output_index);
            }
            TransformerDefinition::StandardScale { mean, scale, .. } => {
                operations.push(Transformer::StandardScale(StandardScaleTransformer::new(
                    input_index,
                    output_index,
                    *mean,
                    *scale,
                )));
            }
        }
    }
    // Operations read only the preceding layout and own distinct output cells, so groups
    // can run after scalar operations without changing authored output order.
    for (input_index, (windows, output_indices)) in lagged_outputs {
        operations.push(Transformer::Lagged(LaggedFeature::new(
            input_index,
            windows,
            output_indices.into_boxed_slice(),
        )));
    }
    for ((ema, input, warmup), (windows, outputs)) in averages {
        operations.push(Transformer::Average(Box::new(AverageTransformer::new(
            ema,
            input,
            &windows,
            outputs.into_boxed_slice(),
            warmup,
        ))));
    }
    operations.into_boxed_slice()
}

/// Resolved scalar operation for allocation-free writes into model input.
pub(crate) enum Transformer {
    Average(Box<AverageTransformer>),
    Identity(IdentityTransformer),
    Lagged(LaggedFeature),
    StandardScale(StandardScaleTransformer),
}

impl Transformer {
    pub(crate) fn apply<V: FeatureVector>(
        &mut self,
        raw_values: &[f64],
        observed: &[bool],
        model_vector: &mut V,
        output_observed: &mut [bool],
    ) {
        match self {
            Self::Average(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
            Self::Identity(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
            Self::Lagged(transformer) => {
                transformer.apply_observed(raw_values, model_vector, output_observed)
            }
            Self::StandardScale(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
        }
    }
}

//! Defines, validates, and executes scalar transformations for model input.
//!
//! Compilation resolves each transformation to indexes and precomputed parameters
//! so execution writes directly into caller-owned storage without allocation.

use std::collections::HashMap;

use crate::{FeatureExtractor, FeatureId, FeatureVector, InvalidTransformationDefinitionError};

mod identity;
mod lagged;
mod standard_scale;

use identity::IdentityTransformer;
use lagged::LaggedFeature;
use standard_scale::StandardScaleTransformer;

const WINDOW_MAX_SIZE: usize = 10_000;

/// One named scalar transformation from the raw feature layout to model input.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerDefinition {
    /// Copies one raw scalar without changing its value.
    Identity { input: FeatureId, output: FeatureId },
    /// Emits the raw scalar from `lag_window` accepted events earlier.
    /// The window must be in `1..=10_000`; output remains NaN until enough history exists.
    /// Definitions for the same input share one runtime history buffer.
    Lagged {
        input: FeatureId,
        output: FeatureId,
        lag_window: usize,
    },
    /// Applies `(input - mean) / scale` to one raw scalar.
    StandardScale {
        input: FeatureId,
        output: FeatureId,
        mean: f64,
        scale: f64,
    },
}

impl TransformerDefinition {
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
            Self::Identity { input, .. }
            | Self::Lagged { input, .. }
            | Self::StandardScale { input, .. } => input,
        }
    }

    pub(crate) fn output(&self) -> &FeatureId {
        match self {
            Self::Identity { output, .. }
            | Self::Lagged { output, .. }
            | Self::StandardScale { output, .. } => output,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), InvalidTransformationDefinitionError> {
        if let Self::Lagged { lag_window, .. } = self {
            if *lag_window == 0 {
                return Err(InvalidTransformationDefinitionError::LagWindowZero);
            }
            if *lag_window > WINDOW_MAX_SIZE {
                return Err(InvalidTransformationDefinitionError::LagWindowTooLarge);
            }
        }
        if let Self::StandardScale { mean, scale, .. } = self {
            if !mean.is_finite() {
                return Err(InvalidTransformationDefinitionError::MeanNotFinite);
            }
            if !scale.is_finite() {
                return Err(InvalidTransformationDefinitionError::ScaleNotFinite);
            }
            if *scale <= 0.0 {
                return Err(InvalidTransformationDefinitionError::ScaleNotPositive);
            }
            if !(1.0 / *scale).is_finite() {
                return Err(InvalidTransformationDefinitionError::InverseScaleNotFinite);
            }
        }
        Ok(())
    }
}

/// Compiles validated scalar definitions, sharing history between lags of one input.
pub(crate) fn compile<V: FeatureVector>(
    definitions: &[TransformerDefinition],
    feature_extractor: &FeatureExtractor<V>,
) -> Box<[Transformer]> {
    let mut operations = Vec::with_capacity(definitions.len());
    let mut lagged_outputs = HashMap::<usize, (Vec<usize>, Vec<usize>)>::new();
    for (output_index, definition) in definitions.iter().enumerate() {
        let input_index = feature_extractor
            .feature_index(definition.input())
            .expect("pipeline construction validated every raw input ID");
        match definition {
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
    // Operations read only raw values and own distinct output cells, so groups
    // can run after scalar operations without changing authored output order.
    for (input_index, (windows, output_indices)) in lagged_outputs {
        operations.push(Transformer::Lagged(LaggedFeature::new(
            input_index,
            windows,
            output_indices.into_boxed_slice(),
        )));
    }
    operations.into_boxed_slice()
}

/// Resolved scalar operation for allocation-free writes into model input.
pub(crate) enum Transformer {
    Identity(IdentityTransformer),
    Lagged(LaggedFeature),
    StandardScale(StandardScaleTransformer),
}

impl Transformer {
    pub(crate) fn apply<V: FeatureVector>(&mut self, raw_values: &[f64], model_vector: &mut V) {
        match self {
            Self::Identity(transformer) => transformer.apply(raw_values, model_vector),
            Self::Lagged(transformer) => transformer.apply(raw_values, model_vector),
            Self::StandardScale(transformer) => transformer.apply(raw_values, model_vector),
        }
    }
}

//! Defines, validates, and executes scalar transformations for model input.
//!
//! Compilation resolves each transformation to indexes and precomputed parameters
//! so execution writes directly into caller-owned storage without allocation.

use crate::{FeatureId, FeatureVector, InvalidTransformationDefinitionError};

mod identity;
mod lagged;
mod standard_scale;

use identity::IdentityTransformer;
use lagged::LaggedFeature;
use standard_scale::StandardScaleTransformer;

/// One named scalar transformation from the raw feature layout to model input.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerDefinition {
    /// Copies one raw scalar without changing its value.
    Identity { input: FeatureId, output: FeatureId },
    /// Emits the raw scalar from `lag_window` accepted events earlier.
    /// The window must be positive; output remains NaN until enough history exists.
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

    pub(super) fn input(&self) -> &FeatureId {
        match self {
            Self::Identity { input, .. }
            | Self::Lagged { input, .. }
            | Self::StandardScale { input, .. } => input,
        }
    }

    pub(super) fn output(&self) -> &FeatureId {
        match self {
            Self::Identity { output, .. }
            | Self::Lagged { output, .. }
            | Self::StandardScale { output, .. } => output,
        }
    }

    pub(super) fn validate(&self) -> Result<(), InvalidTransformationDefinitionError> {
        if matches!(self, Self::Lagged { lag_window: 0, .. }) {
            return Err(InvalidTransformationDefinitionError::LagWindowZero);
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

    /// Compiles a validated definition using indexes resolved by the pipeline.
    pub(super) fn compile(&self, input_index: usize, output_index: usize) -> Transformer {
        match self {
            Self::Identity { .. } => {
                Transformer::Identity(IdentityTransformer::new(input_index, output_index))
            }
            Self::Lagged { lag_window, .. } => {
                Transformer::Lagged(LaggedFeature::new(*lag_window, input_index, output_index))
            }
            Self::StandardScale { mean, scale, .. } => Transformer::StandardScale(
                StandardScaleTransformer::new(input_index, output_index, *mean, *scale),
            ),
        }
    }
}

/// Resolved scalar operation for allocation-free writes into model input.
pub(super) enum Transformer {
    Identity(IdentityTransformer),
    Lagged(LaggedFeature),
    StandardScale(StandardScaleTransformer),
}

impl Transformer {
    pub(super) fn apply<V: FeatureVector>(&mut self, raw_values: &[f64], model_vector: &mut V) {
        match self {
            Self::Identity(transformer) => transformer.apply(raw_values, model_vector),
            Self::Lagged(transformer) => transformer.apply(raw_values, model_vector),
            Self::StandardScale(transformer) => transformer.apply(raw_values, model_vector),
        }
    }
}

use super::{Pipeline, ScalarOperation};
use crate::{
    FeatureId, FeatureVector, FeatureVectorSpec, FimlError, Float, InvalidArgumentError,
    InvalidTransformationDefinitionError, Result,
};

/// One named scalar transformation from the raw feature layout to model input.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformationDefinition<F> {
    /// Copies one raw scalar without changing its value.
    Identity { input: FeatureId, output: FeatureId },
    /// Applies `(input - mean) / scale` to one raw scalar.
    StandardScale {
        input: FeatureId,
        output: FeatureId,
        mean: F,
        scale: F,
    },
}

impl<F> TransformationDefinition<F> {
    /// Creates a scalar identity transformation.
    pub fn identity(input: FeatureId, output: FeatureId) -> Self {
        Self::Identity { input, output }
    }

    /// Creates a scalar standard-scaling transformation.
    pub fn standard_scale(input: FeatureId, output: FeatureId, mean: F, scale: F) -> Self {
        Self::StandardScale {
            input,
            output,
            mean,
            scale,
        }
    }

    fn input(&self) -> &FeatureId {
        match self {
            Self::Identity { input, .. } | Self::StandardScale { input, .. } => input,
        }
    }

    fn output(&self) -> &FeatureId {
        match self {
            Self::Identity { output, .. } | Self::StandardScale { output, .. } => output,
        }
    }
}

/// Validated configuration for raw extraction and the final model-input layout.
///
/// Transformations remain in authored order, which is also final vector order.
/// Raw and final IDs occupy separate layouts and may therefore use the same name.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInputSpec<F> {
    raw_feature_vector_spec: FeatureVectorSpec,
    transformation_definitions: Vec<TransformationDefinition<F>>,
    feature_vector_capacity: usize,
    checksum: Option<String>,
}

impl<F: Float> ModelInputSpec<F> {
    /// Creates a spec whose final width equals its transformation count.
    pub fn new(
        raw_feature_vector_spec: FeatureVectorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformationDefinition<F>>,
    ) -> Result<Self> {
        let transformation_definitions = transformation_definitions.into_iter().collect::<Vec<_>>();
        let capacity = transformation_definitions.len();
        Self::with_metadata(
            raw_feature_vector_spec,
            transformation_definitions,
            capacity,
            None,
        )
    }

    /// Creates a spec with explicit final width and trailing reserved cells.
    pub fn with_capacity(
        raw_feature_vector_spec: FeatureVectorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformationDefinition<F>>,
        feature_vector_capacity: usize,
    ) -> Result<Self> {
        Self::with_metadata(
            raw_feature_vector_spec,
            transformation_definitions,
            feature_vector_capacity,
            None,
        )
    }

    /// Creates a spec with explicit final width and opaque checksum metadata.
    pub fn with_metadata(
        raw_feature_vector_spec: FeatureVectorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformationDefinition<F>>,
        feature_vector_capacity: usize,
        checksum: Option<String>,
    ) -> Result<Self> {
        let transformation_definitions = transformation_definitions.into_iter().collect::<Vec<_>>();
        if feature_vector_capacity < transformation_definitions.len() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::FeatureVectorCapacityTooSmall {
                    capacity: feature_vector_capacity,
                    active_length: transformation_definitions.len(),
                },
            ));
        }

        for (index, definition) in transformation_definitions.iter().enumerate() {
            if !raw_feature_vector_spec
                .definitions()
                .iter()
                .any(|raw| raw.id == *definition.input())
            {
                return invalid_definition(
                    index,
                    InvalidTransformationDefinitionError::InputFeatureNotFound,
                );
            }
            if crate::features::is_reserved_feature_id(definition.output()) {
                return invalid_definition(
                    index,
                    InvalidTransformationDefinitionError::ReservedOutputFeature,
                );
            }
            if transformation_definitions[..index]
                .iter()
                .any(|previous| previous.output() == definition.output())
            {
                return invalid_definition(
                    index,
                    InvalidTransformationDefinitionError::DuplicateOutputFeature,
                );
            }
            if let TransformationDefinition::StandardScale { mean, scale, .. } = definition {
                if !mean.is_finite() {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::MeanNotFinite,
                    );
                }
                if !scale.is_finite() {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::ScaleNotFinite,
                    );
                }
                if *scale <= F::ZERO {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::ScaleNotPositive,
                    );
                }
                if !(F::ONE / *scale).is_finite() {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::InverseScaleNotFinite,
                    );
                }
            }
        }

        Ok(Self {
            raw_feature_vector_spec,
            transformation_definitions,
            feature_vector_capacity,
            checksum,
        })
    }

    /// Returns the raw feature-extraction configuration.
    pub fn raw_feature_vector_spec(&self) -> &FeatureVectorSpec {
        &self.raw_feature_vector_spec
    }

    /// Returns scalar transformations in final model-vector order.
    pub fn transformation_definitions(&self) -> &[TransformationDefinition<F>] {
        &self.transformation_definitions
    }

    /// Returns the complete final width, including reserved cells.
    pub fn feature_vector_capacity(&self) -> usize {
        self.feature_vector_capacity
    }

    /// Returns the number of active final outputs.
    pub fn feature_vector_length(&self) -> usize {
        self.transformation_definitions.len()
    }

    /// Returns opaque checksum metadata without interpreting or verifying it.
    pub fn checksum(&self) -> Option<&str> {
        self.checksum.as_deref()
    }

    /// Compiles this spec into caller-supplied raw and final storage.
    pub fn build<RawV, ModelV>(
        &self,
        raw_vector: RawV,
        mut model_vector: ModelV,
    ) -> Result<Pipeline<F, RawV, ModelV>>
    where
        RawV: FeatureVector<F = F>,
        ModelV: FeatureVector<F = F>,
    {
        if model_vector.capacity() != self.feature_vector_capacity {
            return Err(FimlError::ModelVectorCapacityMismatch {
                expected: self.feature_vector_capacity,
                actual: model_vector.capacity(),
            });
        }
        if model_vector.len() != self.transformation_definitions.len() {
            return Err(FimlError::ModelVectorLengthMismatch {
                expected: self.transformation_definitions.len(),
                actual: model_vector.len(),
            });
        }

        let feature_extractor = self.raw_feature_vector_spec.build(raw_vector)?;
        let mut operations = Vec::with_capacity(self.transformation_definitions.len());
        let mut output_ids = Vec::with_capacity(self.transformation_definitions.len());
        for (output_index, definition) in self.transformation_definitions.iter().enumerate() {
            let input_index = feature_extractor
                .feature_index(definition.input())
                .expect("model-input construction validated every raw input ID");
            let operation = match definition {
                TransformationDefinition::Identity { .. } => ScalarOperation::Identity {
                    input_index,
                    output_index,
                },
                TransformationDefinition::StandardScale { mean, scale, .. } => {
                    ScalarOperation::StandardScale {
                        input_index,
                        output_index,
                        mean: *mean,
                        inverse_scale: F::ONE / *scale,
                    }
                }
            };
            operations.push(operation);
            output_ids.push(definition.output().clone());
        }
        for index in 0..model_vector.capacity() {
            model_vector.set_value_at(index, F::NAN);
        }

        Ok(Pipeline {
            feature_extractor,
            operations: operations.into_boxed_slice(),
            model_vector,
            output_ids: output_ids.into_boxed_slice(),
        })
    }
}

fn invalid_definition<T>(index: usize, reason: InvalidTransformationDefinitionError) -> Result<T> {
    Err(FimlError::InvalidTransformationDefinition { index, reason })
}

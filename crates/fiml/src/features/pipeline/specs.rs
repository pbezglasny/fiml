use super::Pipeline;
use crate::{
    FeatureExtractorSpec, FeatureVector, FimlError, InvalidArgumentError,
    InvalidTransformationDefinitionError, Result, TransformerDefinition,
};

/// Validated configuration for raw extraction and the final model-input layout.
///
/// Transformations remain in authored order, which is also final vector order.
/// Raw and final IDs occupy separate layouts and may therefore use the same name.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineSpec {
    raw_feature_extractor_spec: FeatureExtractorSpec,
    transformation_definitions: Vec<TransformerDefinition>,
    feature_vector_capacity: usize,
    checksum: Option<String>,
}

impl PipelineSpec {
    /// Creates a spec whose final width equals its transformation count.
    pub fn new(
        raw_feature_extractor_spec: FeatureExtractorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformerDefinition>,
    ) -> Result<Self> {
        let transformation_definitions = transformation_definitions.into_iter().collect::<Vec<_>>();
        let capacity = transformation_definitions.len();
        Self::with_metadata(
            raw_feature_extractor_spec,
            transformation_definitions,
            capacity,
            None,
        )
    }

    /// Creates a spec with explicit final width and trailing reserved cells.
    pub fn with_capacity(
        raw_feature_extractor_spec: FeatureExtractorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformerDefinition>,
        feature_vector_capacity: usize,
    ) -> Result<Self> {
        Self::with_metadata(
            raw_feature_extractor_spec,
            transformation_definitions,
            feature_vector_capacity,
            None,
        )
    }

    /// Creates a spec with explicit final width and opaque checksum metadata.
    pub fn with_metadata(
        raw_feature_extractor_spec: FeatureExtractorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformerDefinition>,
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
            if !raw_feature_extractor_spec
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
            definition
                .validate()
                .map_err(|reason| FimlError::InvalidTransformationDefinition { index, reason })?;
        }

        Ok(Self {
            raw_feature_extractor_spec,
            transformation_definitions,
            feature_vector_capacity,
            checksum,
        })
    }

    /// Returns the raw feature-extraction configuration.
    pub fn raw_feature_extractor_spec(&self) -> &FeatureExtractorSpec {
        &self.raw_feature_extractor_spec
    }

    /// Returns scalar transformations in final model-vector order.
    pub fn transformation_definitions(&self) -> &[TransformerDefinition] {
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
    ) -> Result<Pipeline<RawV, ModelV>>
    where
        RawV: FeatureVector,
        ModelV: FeatureVector,
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

        let feature_extractor = self.raw_feature_extractor_spec.build(raw_vector)?;
        let operations = crate::features::transformers::compile(
            &self.transformation_definitions,
            &feature_extractor,
        );
        let output_ids = self
            .transformation_definitions
            .iter()
            .map(|definition| definition.output().clone())
            .collect();
        for index in 0..model_vector.capacity() {
            model_vector.set_value_at(index, f64::NAN);
        }

        Ok(Pipeline {
            feature_extractor,
            operations,
            model_vector,
            output_ids,
        })
    }
}

fn invalid_definition<T>(index: usize, reason: InvalidTransformationDefinitionError) -> Result<T> {
    Err(FimlError::InvalidTransformationDefinition { index, reason })
}

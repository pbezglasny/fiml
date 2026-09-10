use super::{FittedStage, Pipeline, StageRuntime};
use crate::{
    FeatureExtractorSpec, FeatureVector, FimlError, InvalidArgumentError, Result,
    TransformerDefinition,
};

/// Validated configuration for raw extraction and the final model-input layout.
///
/// Scalar transformations define the base order; each fitted stage consumes that
/// layout or the preceding stage, as do additional scalar stages.
/// The last stage owns the final vector layout.
/// Layouts have separate ID namespaces and may therefore reuse names.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineSpec {
    raw_feature_extractor_spec: FeatureExtractorSpec,
    transformation_definitions: Vec<TransformerDefinition>,
    stages: Vec<FittedStage>,
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
        Self::with_stages(
            raw_feature_extractor_spec,
            transformation_definitions,
            [],
            feature_vector_capacity,
            checksum,
        )
    }

    /// Validates a base scalar layout followed by scalar or fitted vector stages.
    /// Capacity describes the final output, independently of intermediate widths.
    pub fn with_stages(
        raw_feature_extractor_spec: FeatureExtractorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformerDefinition>,
        stages: impl IntoIterator<Item = FittedStage>,
        feature_vector_capacity: usize,
        checksum: Option<String>,
    ) -> Result<Self> {
        let transformation_definitions = transformation_definitions.into_iter().collect::<Vec<_>>();
        let stages = stages.into_iter().collect::<Vec<_>>();
        let final_length = stages
            .last()
            .map_or(transformation_definitions.len(), |stage| {
                stage.outputs().len()
            });
        if feature_vector_capacity < final_length {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::FeatureVectorCapacityTooSmall {
                    capacity: feature_vector_capacity,
                    active_length: final_length,
                },
            ));
        }

        let raw_ids = raw_feature_extractor_spec
            .definitions()
            .iter()
            .map(|definition| definition.id.clone())
            .collect::<Vec<_>>();
        crate::features::transformers::validate(&transformation_definitions, &raw_ids)?;

        if !stages.is_empty() {
            let base_ids = transformation_definitions
                .iter()
                .map(|definition| definition.output().clone())
                .collect::<Vec<_>>();
            let mut inputs = std::borrow::Cow::Borrowed(base_ids.as_slice());
            for (index, stage) in stages.iter().enumerate() {
                stage
                    .validate(&inputs)
                    .map_err(|reason| FimlError::InvalidPipelineStage { index, reason })?;
                inputs = stage.outputs();
            }
        }

        Ok(Self {
            raw_feature_extractor_spec,
            transformation_definitions,
            stages,
            feature_vector_capacity,
            checksum,
        })
    }

    /// Returns the raw feature-extraction configuration.
    pub fn raw_feature_extractor_spec(&self) -> &FeatureExtractorSpec {
        &self.raw_feature_extractor_spec
    }

    /// Returns scalar transformations in base-vector order.
    pub fn transformation_definitions(&self) -> &[TransformerDefinition] {
        &self.transformation_definitions
    }

    /// Returns the complete final width, including reserved cells.
    pub fn feature_vector_capacity(&self) -> usize {
        self.feature_vector_capacity
    }

    /// Returns the number of active final outputs.
    pub fn feature_vector_length(&self) -> usize {
        self.stages
            .last()
            .map_or(self.transformation_definitions.len(), |stage| {
                stage.outputs().len()
            })
    }

    /// Scalar and fitted vector stages in execution order.
    pub fn stages(&self) -> &[FittedStage] {
        &self.stages
    }

    /// Final active IDs, resolved on the cold configuration path.
    pub fn output_ids(&self) -> Vec<crate::FeatureId> {
        match self.stages.last() {
            Some(stage) => stage.outputs().to_vec(),
            None => self
                .transformation_definitions
                .iter()
                .map(|definition| definition.output().clone())
                .collect(),
        }
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
        if model_vector.len() != self.feature_vector_length() {
            return Err(FimlError::ModelVectorLengthMismatch {
                expected: self.feature_vector_length(),
                actual: model_vector.len(),
            });
        }

        let feature_extractor = self.raw_feature_extractor_spec.build(raw_vector)?;
        let operations = crate::features::transformers::compile(
            &self.transformation_definitions,
            feature_extractor.feature_ids(),
        );
        let output_ids = self.output_ids().into_boxed_slice();
        for index in 0..model_vector.capacity() {
            model_vector.set_value_at(index, f64::NAN);
        }

        Ok(Pipeline {
            feature_extractor,
            operations,
            stages: StageRuntime::new(&self.transformation_definitions, &self.stages),
            model_vector,
            output_ids,
        })
    }
}

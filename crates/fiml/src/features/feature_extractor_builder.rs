use crate::features::compiler;
use crate::{FeatureDefinition, FeatureExtractor, FeatureVector, FimlError, Float};

pub struct FeatureExtractorBuilder<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    definitions: Vec<FeatureDefinition>,
    output_vector: V,
}

impl<F, V> FeatureExtractorBuilder<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new(output_vector: V) -> Self {
        Self {
            definitions: Vec::new(),
            output_vector,
        }
    }

    pub fn add_feature(mut self, feature_definition: FeatureDefinition) -> Self {
        self.definitions.push(feature_definition);
        self
    }

    pub fn build(self) -> Result<FeatureExtractor<F, V>, FimlError> {
        let compilation = compiler::compile(self.definitions, self.output_vector.len())?;
        Ok(FeatureExtractor::new(self.output_vector, compilation))
    }
}

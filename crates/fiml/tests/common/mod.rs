//! Shared single-branch fixtures for transformation tests.
use fiml::{FeatureExtractorSpec, FittedStage, PipelineSpec, Result, TransformerDefinition};

pub fn linear_spec(
    raw: FeatureExtractorSpec,
    transformations: impl IntoIterator<Item = TransformerDefinition>,
    stages: impl IntoIterator<Item = FittedStage>,
    checksum: Option<String>,
) -> Result<PipelineSpec> {
    let mut spec = PipelineSpec::new(raw);
    spec.set_checksum(checksum);
    let transformations = transformations.into_iter().collect::<Vec<_>>();
    let mut node = spec.root();
    if !transformations.is_empty() {
        node = spec.add(node, FittedStage::Scalar { transformations })?;
    }
    for stage in stages {
        node = spec.add(node, stage)?;
    }
    spec.output("model", node)?;
    spec.validate()?;
    Ok(spec)
}

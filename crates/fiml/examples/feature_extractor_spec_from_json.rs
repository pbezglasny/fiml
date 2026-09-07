use fiml::{FeatureExtractorSpec, VecFeatureVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = include_str!("../../../notebooks/feature_extractor_spec.json");
    let feature_extractor_spec: FeatureExtractorSpec = serde_json::from_str(json)?;
    let output = VecFeatureVector::new_of_length(
        feature_extractor_spec.feature_vector_capacity(),
        feature_extractor_spec.feature_vector_length(),
    );
    let extractor = feature_extractor_spec.build(output)?;

    println!(
        "loaded {} active features into {} model cells",
        extractor.feature_ids().len(),
        feature_extractor_spec.feature_vector_capacity()
    );
    Ok(())
}

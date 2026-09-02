use fiml::{FeatureVectorSpec, VecFeatureVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = include_str!("../../../notebooks/feature_vector_spec.json");
    let feature_vector_spec: FeatureVectorSpec = serde_json::from_str(json)?;
    let output = VecFeatureVector::new_of_length(
        feature_vector_spec.feature_vector_capacity(),
        feature_vector_spec.feature_vector_length(),
    );
    let extractor = feature_vector_spec.build(output)?;

    println!(
        "loaded {} active features into {} model cells",
        extractor.feature_ids().len(),
        feature_vector_spec.feature_vector_capacity()
    );
    Ok(())
}

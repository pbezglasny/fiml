use fiml::{FeatureSet, VecFeatureVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = include_str!("../../../notebooks/feature_set.json");
    let feature_set: FeatureSet = serde_json::from_str(json)?;
    let output = VecFeatureVector::<f64>::new_of_length(
        feature_set.feature_vector_capacity(),
        feature_set.feature_vector_length(),
    );
    let extractor = feature_set.build(output)?;

    println!(
        "loaded {} active features into {} model cells",
        extractor.feature_ids().len(),
        feature_set.feature_vector_capacity()
    );
    Ok(())
}

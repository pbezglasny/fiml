//! Load a standalone `FeatureSet` JSON artifact produced by Python.
//!
//! From the repository root:
//!
//! ```text
//! cargo run -p fiml --example feature_set_from_json --features serde
//! ```
//!
//! Pass another JSON path as the first argument to load a different artifact.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use fiml::{Event, FeatureExtractor, FeatureSet, IndicatorFeatures, symbols};

fn default_feature_set_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../notebooks/feature_set.json")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_feature_set_path)
        .canonicalize()?;
    let json = fs::read_to_string(&path)?;
    let feature_set: FeatureSet = serde_json::from_str(&json)?;
    let mut extractor = FeatureExtractor::from_feature_set(&feature_set)?;

    println!("loaded: {}", path.display());
    println!("indicators: {}", feature_set.indicator_count());
    println!("outputs: {}", feature_set.output_count());
    println!("columns: {:?}", extractor.feature_names());

    let symbol_names: BTreeSet<_> = feature_set
        .indicators
        .iter()
        .filter_map(|definition| definition.symbol.as_deref())
        .collect();
    for (index, symbol_name) in symbol_names.into_iter().enumerate() {
        let timestamp = (index + 1) as i64;
        let symbol = symbols::intern(symbol_name);
        extractor.dispatch(&Event::trade(symbol, 100.0, 1.0, timestamp, None))?;
    }

    println!("values: {:?}", extractor.values());
    Ok(())
}

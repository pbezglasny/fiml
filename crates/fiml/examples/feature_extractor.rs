//! Live/serving side of the Python <-> Rust parity story.
//!
//! Builds a feature extractor from the same `FeatureSet` JSON that Python
//! training uses (see `crates/fiml-python/examples/quickstart.py`) and
//! dispatches the same price series. The printed feature values match what
//! `FeatureExtractor.transform` produces in Python, because both run this
//! exact extractor.
//!
//! Run with: `cargo run --example feature_extractor --features serde`

use fiml::{Event, FeatureExtractor, FeatureSet, IndicatorFeatures, symbols};

const FEATURE_SET_JSON: &str = r#"{
    "version": "1.0.0",
    "indicators": [
        {
            "symbol": "BTCUSDT",
            "indicator": { "Sma": { "source": "price", "windows": [3] } }
        },
        {
            "symbol": "BTCUSDT",
            "indicator": { "Ema": { "source": "price", "windows": [3] } }
        }
    ]
}"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let feature_set: FeatureSet = serde_json::from_str(FEATURE_SET_JSON)?;
    let mut extractor = FeatureExtractor::from_feature_set(&feature_set)?;

    let btc = symbols::intern("BTCUSDT");
    let prices = [10.0, 11.0, 9.0, 12.0, 13.0, 12.5];

    println!("columns: {:?}", extractor.feature_names());
    for (timestamp, price) in prices.iter().enumerate() {
        extractor.dispatch(&Event::price(btc, *price, timestamp as i64))?;
        println!("t={timestamp} price={price} -> {:?}", extractor.values());
    }

    Ok(())
}

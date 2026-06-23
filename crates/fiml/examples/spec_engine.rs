//! Live/serving side of the Python <-> Rust parity story.
//!
//! Builds an engine from the same `EngineSpec` JSON that Python training uses
//! (see `fiml-python/examples/quickstart.py`) and dispatches the same price
//! series. The printed feature values match what `Engine.transform` produces in
//! Python, because both run this exact engine.
//!
//! Run with: `cargo run --example spec_engine --features serde`

use fiml::{DynIndicatorEngine, EngineSpec, Event, IndicatorFeatures, symbols};

const SPEC_JSON: &str = r#"{
    "features": [
        { "name": "sma_3", "symbol": "BTCUSDT", "spec": { "Sma": { "period": 3 } } },
        { "name": "ema_3", "symbol": "BTCUSDT", "spec": { "Ema": { "period": 3 } } }
    ]
}"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec: EngineSpec = serde_json::from_str(SPEC_JSON)?;
    let mut engine = DynIndicatorEngine::from_spec(&spec)?;

    let btc = symbols::intern("BTCUSDT");
    let prices = [10.0, 11.0, 9.0, 12.0, 13.0, 12.5];

    println!("columns: {:?}", engine.feature_names());
    for (timestamp, price) in prices.iter().enumerate() {
        engine.dispatch(&Event::price(btc, *price, timestamp as i64));
        println!("t={timestamp} price={price} -> {:?}", engine.values());
    }

    Ok(())
}

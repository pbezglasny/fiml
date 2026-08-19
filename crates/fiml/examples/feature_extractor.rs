//! Build an extractor from scalar feature definitions and process price ticks.

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractor, FeatureKey,
    FeatureSource, FeatureVector, WarmupPolicy, symbols,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let btc = symbols::intern("BTCUSDT");
    let source = FeatureSource::Field(EventField::Price);
    let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<f64, 2>::new())
        .add_feature(FeatureDefinition::with_default_id(FeatureKey::Ema {
            symbol: btc,
            source,
            window: 3,
            warmup_policy: WarmupPolicy::FullWindow,
        }))
        .add_feature(FeatureDefinition::with_default_id(FeatureKey::Sma {
            symbol: btc,
            source,
            window: 3,
            warmup_policy: WarmupPolicy::FullWindow,
        }))
        .build()?;
    let prices = [10.0, 11.0, 9.0, 12.0, 13.0, 12.5];

    let columns: Vec<_> = extractor
        .feature_ids()
        .iter()
        .map(|id| id.as_str())
        .collect();
    println!("columns: {columns:?}");
    for (timestamp, price) in prices.iter().enumerate() {
        extractor.handle_event(&Event::price(btc, *price, timestamp as i64))?;
        println!(
            "t={timestamp} price={price} -> {:?}",
            extractor.feature_vector().values()
        );
    }

    Ok(())
}

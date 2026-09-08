//! Derive price and order-book features into the same output vector.

use fiml::order_book::{
    OrderBook, OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate, OrderBookSnapshot, Side,
    UpdatePolicy,
};
use rust_decimal::dec;

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractor, FeatureKey,
    FeatureSource, FeatureVector, WarmupPolicy, symbols,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let btc = symbols::intern("BTCUSDT")?;
    let source = FeatureSource::Field(EventField::Price);
    let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<5>::new())
        .add_order_book(btc, OrderBook::new(UpdatePolicy::Contiguous, 8))
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
        .add_feature(FeatureDefinition::with_default_id(
            FeatureKey::OrderBookMidPrice { symbol: btc },
        ))
        .add_feature(FeatureDefinition::with_default_id(
            FeatureKey::OrderBookImbalance {
                symbol: btc,
                n_levels: 1,
            },
        ))
        .add_feature(FeatureDefinition::with_default_id(
            FeatureKey::OrderBookImbalance {
                symbol: btc,
                n_levels: 5,
            },
        ))
        .build()?;
    let prices = [10.0, 11.0, 9.0, 12.0, 13.0, 12.5];

    let columns: Vec<_> = extractor
        .feature_ids()
        .iter()
        .map(|id| id.as_str())
        .collect();
    println!("columns: {columns:?}");
    for (timestamp, price) in prices.iter().enumerate() {
        extractor.handle_event(Event::price(btc, *price, timestamp as i64))?;
        println!(
            "t={timestamp} price={price} -> {:?}",
            extractor.feature_vector().values()
        );
    }

    extractor.handle_event(Event::order_book_snapshot(
        btc,
        6,
        OrderBookSnapshot::new(
            10,
            vec![
                OrderBookLevel::new(dec!(12), dec!(2)),
                OrderBookLevel::new(dec!(11), dec!(4)),
            ],
            vec![OrderBookLevel::new(dec!(13), dec!(3))],
        ),
    ))?;
    println!("snapshot -> {:?}", extractor.feature_vector().values());
    extractor.handle_event(Event::order_book_delta(
        btc,
        7,
        OrderBookDelta::new(
            11,
            vec![OrderBookLevelUpdate::new(Side::Bid, dec!(12), dec!(5))],
        ),
    ))?;
    println!("delta -> {:?}", extractor.feature_vector().values());

    Ok(())
}

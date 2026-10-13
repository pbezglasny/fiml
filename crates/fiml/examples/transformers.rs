//! Extract market state at multiple scales using changes, spread std, and imbalance range.
//! Run with `cargo run -p fiml --example transformers`.

use fiml::order_book::{OrderBookConfig, OrderBookLevel, OrderBookSnapshot, UpdatePolicy};
use fiml::{
    ArrayFeatureVector, Event, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey,
    FittedStage, PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};
use rust_decimal::Decimal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let btc = Symbol::new("BTCUSDT")?;
    let id = FeatureId::new;
    let raw = FeatureExtractorSpec::new([
        FeatureDefinition::new(FeatureKey::OrderBookSpread { symbol: btc }, id("spread")),
        FeatureDefinition::new(
            FeatureKey::OrderBookImbalance {
                symbol: btc,
                n_levels: 1,
            },
            id("imbalance"),
        ),
    ])?
    .with_order_books([OrderBookConfig::new(btc, UpdatePolicy::Contiguous, 8)])?;

    // Each stage reads the preceding layout; definition order sets model columns.
    let spec = PipelineSpec::with_stages(
        raw,
        [
            T::standard_scale(id("spread"), id("scaled_spread"), 0., 2.),
            T::identity(id("imbalance"), id("imbalance")),
        ],
        [FittedStage::Scalar {
            transformations: vec![
                // All four outputs share one finite-observation history.
                T::delta(id("scaled_spread"), id("delta1"), 1),
                T::simple_return(id("scaled_spread"), id("return1"), 1),
                T::log_return(id("scaled_spread"), id("log_return1"), 1),
                T::delta(id("scaled_spread"), id("delta2"), 2),
                T::rolling_std(
                    id("scaled_spread"),
                    id("spread_std2"),
                    2,
                    WarmupPolicy::FirstValue,
                ),
                T::rolling_std(
                    id("scaled_spread"),
                    id("spread_std4"),
                    4,
                    WarmupPolicy::FirstValue,
                ),
                T::rolling_range(
                    id("imbalance"),
                    id("imbalance_range2"),
                    2,
                    WarmupPolicy::FirstValue,
                ),
                T::rolling_range(
                    id("imbalance"),
                    id("imbalance_range4"),
                    4,
                    WarmupPolicy::FirstValue,
                ),
            ],
        }],
        8,
        None,
    )?;
    let mut pipeline = spec.build(
        ArrayFeatureVector::<2>::new(),
        ArrayFeatureVector::<8>::new(),
    )?;
    println!("model columns: {:?}", pipeline.output_ids());
    for (index, ask) in [102, 104, 108, 108].into_iter().enumerate() {
        pipeline.handle_event(Event::order_book_snapshot(
            btc,
            index as i64,
            OrderBookSnapshot::new(
                index as u64,
                vec![OrderBookLevel::new(
                    Decimal::from(100),
                    Decimal::from(index + 1),
                )],
                vec![OrderBookLevel::new(Decimal::from(ask), Decimal::ONE)],
            ),
        ))?;
        println!(
            "market={:?} model={:?}",
            pipeline.raw_values(),
            pipeline.values()
        );
        // Unrelated events do not add duplicate spread observations.
        pipeline.handle_event(Event::time(index as i64))?;
        assert_eq!(pipeline.output_observations(), &[false; 8]);
    }
    assert_eq!(pipeline.raw_values(), &[8., 0.6]);
    let expected = [0., 0., 0., 2., 0., 1.6875_f64.sqrt(), 0.1, 0.6];
    for (actual, expected) in pipeline.values().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    Ok(())
}

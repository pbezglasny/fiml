//! Extract an order-book spread, scale it, and compute observation-based changes.
//! Run with `cargo run -p fiml --example transformers`.

use fiml::order_book::{OrderBookConfig, OrderBookLevel, OrderBookSnapshot, UpdatePolicy};
use fiml::{
    ArrayFeatureVector, Event, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey,
    FittedStage, PipelineSpec, Symbol, TransformerDefinition as T,
};
use rust_decimal::Decimal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let btc = Symbol::new("BTCUSDT")?;
    let id = FeatureId::new;
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::OrderBookSpread { symbol: btc },
        id("spread"),
    )])?
    .with_order_books([OrderBookConfig::new(btc, UpdatePolicy::Contiguous, 8)])?;

    // Each stage reads the preceding layout; definition order sets model columns.
    let spec = PipelineSpec::with_stages(
        raw,
        [T::standard_scale(id("spread"), id("scaled_spread"), 0., 2.)],
        [FittedStage::Scalar {
            transformations: vec![
                // All four outputs share one finite-observation history.
                T::delta(id("scaled_spread"), id("delta1"), 1),
                T::simple_return(id("scaled_spread"), id("return1"), 1),
                T::log_return(id("scaled_spread"), id("log_return1"), 1),
                T::delta(id("scaled_spread"), id("delta2"), 2),
            ],
        }],
        4,
        None,
    )?;
    let mut pipeline = spec.build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<4>::new(),
    )?;
    println!("model columns: {:?}", pipeline.output_ids());
    for (index, ask) in [102, 104, 108, 108].into_iter().enumerate() {
        pipeline.handle_event(Event::order_book_snapshot(
            btc,
            index as i64,
            OrderBookSnapshot::new(
                index as u64,
                vec![OrderBookLevel::new(Decimal::from(100), Decimal::ONE)],
                vec![OrderBookLevel::new(Decimal::from(ask), Decimal::ONE)],
            ),
        ))?;
        println!(
            "spread={:?} changes={:?}",
            pipeline.raw_values(),
            pipeline.values()
        );
        // Unrelated events do not add duplicate spread observations.
        pipeline.handle_event(Event::time(index as i64))?;
        assert_eq!(pipeline.output_observations(), &[false; 4]);
    }
    assert_eq!(pipeline.raw_values(), &[8.]);
    assert_eq!(pipeline.values(), &[0., 0., 0., 2.]);
    Ok(())
}

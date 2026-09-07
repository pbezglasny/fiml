//! Transform a raw SMA into identity, scaled, and event-lagged model inputs.
//! Run with `cargo run -p fiml --example transformers`.

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, FeatureSource, PipelineSpec, TransformerDefinition, WarmupPolicy, symbols,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let btc = symbols::intern("BTCUSDT")?;
    let raw_id = FeatureId::new("raw_sma");
    let raw_spec = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Sma {
            symbol: btc,
            source: FeatureSource::Field(EventField::Price),
            window: 2,
            warmup_policy: WarmupPolicy::FullWindow,
        },
        raw_id.clone(),
    )])?;

    // Every transformer reads the raw layout; definition order sets model columns.
    let model_spec = PipelineSpec::new(
        raw_spec,
        [
            TransformerDefinition::identity(raw_id.clone(), FeatureId::new("sma")),
            // Example fitted parameters: (raw_sma - 10) / 2.
            TransformerDefinition::standard_scale(
                raw_id.clone(),
                FeatureId::new("scaled_sma"),
                10.0,
                2.0,
            ),
            TransformerDefinition::lagged(raw_id, FeatureId::new("sma_lag_2"), 2),
        ],
    )?;
    // Raw and model storage are allocated once, before processing events.
    let mut pipeline = model_spec.build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<3>::new(),
    )?;

    println!("model columns: {:?}", pipeline.output_ids());
    // Lag counts accepted events, including samples where the raw SMA is NaN.
    // The first finite SMA appears on event 2; its lagged value on event 4.
    for (timestamp, price) in [10.0, 12.0, 14.0, 16.0, 18.0, 20.0].into_iter().enumerate() {
        pipeline.handle_event(Event::price(btc, price, timestamp as i64))?;
        println!(
            "t={timestamp} price={price} raw={:?} model={:?}",
            pipeline.raw_values(),
            pipeline.values(),
        );
    }

    assert_eq!(pipeline.raw_values(), &[19.0]);
    assert_eq!(pipeline.values(), &[19.0, 4.5, 15.0]);
    Ok(())
}

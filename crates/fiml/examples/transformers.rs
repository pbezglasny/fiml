//! Extract price, smooth it, and transform the SMA into identity, scaled, and event-lagged model inputs.
//! Run with `cargo run -p fiml --example transformers`.

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, FittedStage, PipelineSpec, Symbol, TransformerDefinition, WarmupPolicy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let btc = Symbol::new("BTCUSDT")?;
    let raw_id = FeatureId::new("price");
    let average_id = FeatureId::new("average");
    let raw_spec = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Field {
            symbol: btc,
            field: EventField::Price,
        },
        raw_id.clone(),
    )])?;

    // Each stage reads the preceding layout; definition order sets model columns.
    let model_spec = PipelineSpec::with_stages(
        raw_spec,
        [TransformerDefinition::sma(
            raw_id,
            average_id.clone(),
            2,
            WarmupPolicy::FullWindow,
        )],
        [FittedStage::Scalar {
            transformations: vec![
                TransformerDefinition::identity(average_id.clone(), FeatureId::new("sma")),
                // Example fitted parameters: (average - 10) / 2.
                TransformerDefinition::standard_scale(
                    average_id.clone(),
                    FeatureId::new("scaled_sma"),
                    10.0,
                    2.0,
                ),
                // These two outputs share one history buffer for average.
                TransformerDefinition::lagged(average_id.clone(), FeatureId::new("sma_lag_2"), 2),
                TransformerDefinition::lagged(average_id, FeatureId::new("sma_lag_1"), 1),
            ],
        }],
        4,
        None,
    )?;
    // Raw and model storage are allocated once, before processing events.
    let mut pipeline = model_spec.build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<4>::new(),
    )?;

    println!("model columns: {:?}", pipeline.output_ids());
    // Lag counts accepted events, including samples where the SMA is NaN.
    // The first finite SMA appears on event 2; lag 1 on event 3 and lag 2 on event 4.
    for (timestamp, price) in [10.0, 12.0, 14.0, 16.0, 18.0, 20.0].into_iter().enumerate() {
        pipeline.handle_event(Event::price(btc, price, timestamp as i64))?;
        println!(
            "t={timestamp} price={price} raw={:?} model={:?}",
            pipeline.raw_values(),
            pipeline.values(),
        );
    }

    assert_eq!(pipeline.raw_values(), &[20.0]);
    assert_eq!(pipeline.values(), &[19.0, 4.5, 15.0, 17.0]);
    Ok(())
}

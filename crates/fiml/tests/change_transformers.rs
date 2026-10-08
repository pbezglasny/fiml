use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, FittedStage, PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};

fn id(name: &str) -> FeatureId {
    FeatureId::new(name)
}

fn raw(symbol: Symbol) -> FeatureExtractorSpec {
    FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Field {
            symbol,
            field: EventField::Price,
        },
        id("p"),
    )])
    .unwrap()
}

fn equal(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() < 1e-12,
            "{actual} != {expected}"
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn shared_artifact_preserves_order_and_observation_lags_across_symbols_and_stages() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/change_transformers.json"
    ))
    .unwrap();
    let spec: PipelineSpec = serde_json::from_value(fixture["pipeline"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), fixture["pipeline"]);
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<2>::new(),
            ArrayFeatureVector::<8>::new(),
        )
        .unwrap();
    assert_eq!(
        pipeline.output_ids(),
        [
            "d3",
            "r1",
            "l1",
            "d1",
            "d1_copy",
            "q_return",
            "raw_delta",
            "event_lag"
        ]
        .map(id)
    );
    for step in fixture["steps"].as_array().unwrap() {
        let event = &step["event"];
        let timestamp = event["timestamp"].as_i64().unwrap();
        let event = if event["kind"] == "time" {
            Event::time(timestamp)
        } else {
            Event::trade(
                Symbol::new(event["symbol"].as_str().unwrap()).unwrap(),
                event["price"].as_f64().unwrap(),
                1.,
                timestamp,
                None,
            )
        };
        pipeline.handle_event(event).unwrap();
        let expected: Vec<_> = step["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN))
            .collect();
        equal(pipeline.values(), &expected);
    }
}

#[test]
fn changes_sample_only_finite_observations_after_sma_and_pca() {
    let symbol = Symbol::new("change-stages").unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw(symbol),
        [T::standard_scale(id("p"), id("x"), 0., 0.5)],
        [
            FittedStage::Scalar {
                transformations: vec![T::sma(id("x"), id("x"), 2, WarmupPolicy::FullWindow)],
            },
            FittedStage::Pca {
                outputs: vec![id("pc")],
                mean: vec![0.],
                components: vec![vec![1.]],
                output_scale: vec![1.],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::delta(id("pc"), id("d"), 1),
                    T::simple_return(id("pc"), id("r"), 1),
                    T::log_return(id("pc"), id("l"), 1),
                ],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::identity(id("d"), id("d")),
                    T::delta(id("d"), id("dd"), 1),
                ],
            },
        ],
        2,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<2>::new(),
    )
    .unwrap();
    for (timestamp, (price, expected)) in [
        (1., [f64::NAN, f64::NAN]),    // SMA warming up.
        (3., [f64::NAN, f64::NAN]),    // First finite SMA = 4.
        (5., [4., f64::NAN]),          // SMA = 8; first delta.
        (1e308, [f64::NAN, f64::NAN]), // Scaling overflows; neither history advances.
        (9., [6., 2.]),                // SMA = 14; delta compares against 8.
        (9., [4., -2.]),               // SMA = 18; independent second-stage history.
    ]
    .into_iter()
    .enumerate()
    {
        pipeline
            .handle_event(Event::price(symbol, price, timestamp as i64))
            .unwrap();
        equal(pipeline.values(), &expected);
        assert_eq!(pipeline.output_observations(), &[true, true]);
        pipeline
            .handle_event(Event::volume(symbol, 1., timestamp as i64))
            .unwrap();
        equal(pipeline.values(), &expected);
        assert_eq!(pipeline.output_observations(), &[false, false]);
    }
}

#[test]
fn validates_lags_and_preserves_history_on_rejected_events_and_context_refresh() {
    let symbol = Symbol::new("change-validation").unwrap();
    for constructor in [T::delta, T::simple_return, T::log_return] {
        for lag in [0, 10_001] {
            assert!(PipelineSpec::new(raw(symbol), [constructor(id("p"), id("x"), lag)]).is_err());
        }
        assert!(PipelineSpec::new(raw(symbol), [constructor(id("p"), id("x"), 10_000)]).is_ok());
        assert!(PipelineSpec::new(raw(symbol), [constructor(id("missing"), id("x"), 1)]).is_err());
    }
    let spec = FeatureExtractorSpec::new([
        raw(symbol).definitions()[0].clone(),
        FeatureDefinition::new(
            FeatureKey::Context {
                symbol,
                name: "context".into(),
            },
            id("context"),
        ),
    ])
    .unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        spec,
        [
            T::delta(id("p"), id("d"), 1),
            T::delta(id("context"), id("c"), 1),
        ],
        [FittedStage::Scalar {
            transformations: vec![T::delta(id("d"), id("dd"), 1)],
        }],
        1,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<2>::new(),
        ArrayFeatureVector::<1>::new(),
    )
    .unwrap();
    for (timestamp, price) in [1., 3., 6.].into_iter().enumerate() {
        pipeline
            .handle_event(Event::price(symbol, price, timestamp as i64))
            .unwrap();
    }
    equal(pipeline.values(), &[1.]);
    pipeline.update_context(&[("context", Some(10.))]).unwrap();
    equal(pipeline.values(), &[1.]);
    assert_eq!(pipeline.output_observations(), &[false]);
    for event in [
        Event::price(symbol, f64::NAN, 3),
        Event::price(symbol, 99., 0),
    ] {
        assert!(pipeline.handle_event(event).is_err());
        equal(pipeline.values(), &[1.]);
    }
    pipeline.handle_event(Event::price(symbol, 10., 3)).unwrap();
    equal(pipeline.values(), &[1.]);
}

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, FittedStage, InvalidTransformationDefinitionError, PipelineSpec, Symbol,
    TransformerDefinition as T,
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
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a == e || a.is_nan() && e.is_nan(),
            "{actual:?} != {expected:?}"
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn shared_artifact_replays_delta_signals_clocks_and_independent_symbols() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/state_transformers.json"
    ))
    .unwrap();
    let spec: PipelineSpec = serde_json::from_value(fixture["pipeline"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), fixture["pipeline"]);
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<2>::new(),
            ArrayFeatureVector::<9>::new(),
        )
        .unwrap();
    let expected_ids: Vec<_> = fixture["pipeline"]["model_input"]["stages"][0]["transformations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| id(t["output"].as_str().unwrap()))
        .collect();
    assert_eq!(pipeline.output_ids(), expected_ids);
    for step in fixture["steps"].as_array().unwrap() {
        let event = &step["event"];
        let timestamp = event["timestamp"].as_i64().unwrap();
        let symbol = Symbol::new(event["symbol"].as_str().unwrap_or("a")).unwrap();
        let event = match event["kind"].as_str().unwrap() {
            "time" => Event::time(timestamp),
            "volume" => Event::volume(symbol, 1., timestamp),
            _ => Event::trade(
                symbol,
                event["price"].as_f64().unwrap(),
                1.,
                timestamp,
                None,
            ),
        };
        pipeline.handle_event(event).unwrap();
        let expected: Vec<_> = step["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect();
        equal(pipeline.values(), &expected);
        let observed: Vec<_> = step["observed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        assert_eq!(pipeline.output_observations(), observed);
    }
}

#[test]
fn validates_epsilon_in_base_and_scalar_stages() {
    for constructor in [
        T::positive_streak,
        T::negative_streak,
        T::observations_since_positive,
        T::observations_since_negative,
        T::observations_since_nonzero,
        T::time_since_positive,
        T::time_since_negative,
        T::time_since_nonzero,
    ] {
        for epsilon in [-1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let definition = constructor(id("p"), id("state"), epsilon);
            assert!(matches!(
                PipelineSpec::new(raw(Symbol::GLOBAL), [definition.clone()]),
                Err(fiml::FimlError::InvalidTransformationDefinition {
                    reason: InvalidTransformationDefinitionError::InvalidEpsilon,
                    ..
                })
            ));
            assert!(
                PipelineSpec::with_stages(
                    raw(Symbol::GLOBAL),
                    [T::identity(id("p"), id("p"))],
                    [FittedStage::Scalar {
                        transformations: vec![definition]
                    }],
                    1,
                    None,
                )
                .is_err()
            );
        }
        for epsilon in [0., -0., 0.5, f64::MAX] {
            assert!(
                PipelineSpec::new(
                    raw(Symbol::GLOBAL),
                    [constructor(id("p"), id("state"), epsilon)]
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn signals_after_delta_and_returns_feed_later_scalar_stages() {
    let symbol = Symbol::new("state-returns").unwrap();
    for change in [T::delta, T::simple_return, T::log_return] {
        let mut pipeline = PipelineSpec::with_stages(
            raw(symbol),
            [change(id("p"), id("signal"), 1)],
            [
                FittedStage::Scalar {
                    transformations: vec![
                        T::positive_streak(id("signal"), id("streak"), 0.),
                        T::time_since_positive(id("signal"), id("age"), 0.),
                    ],
                },
                FittedStage::Scalar {
                    transformations: vec![
                        T::identity(id("streak"), id("streak")),
                        T::delta(id("age"), id("age_change"), 1),
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
        for (event, expected, observed) in [
            (Event::price(symbol, 2., 0), [f64::NAN; 2], [true; 2]),
            (Event::price(symbol, 4., 10), [1., f64::NAN], [true; 2]),
            (Event::volume(symbol, 1., 30), [1., 20.], [false, true]),
            (Event::price(symbol, 8., 40), [2., -20.], [true; 2]),
            (Event::price(symbol, 8., 50), [0., 10.], [true; 2]),
        ] {
            pipeline.handle_event(event).unwrap();
            equal(pipeline.values(), &expected);
            assert_eq!(pipeline.output_observations(), observed);
        }
    }
}

#[test]
fn missing_values_context_refreshes_and_rejected_events_preserve_state() {
    let a = Symbol::new("state-context").unwrap();
    let raw = FeatureExtractorSpec::new([
        raw(a).definitions()[0].clone(),
        FeatureDefinition::new(
            FeatureKey::Context {
                symbol: a,
                name: "context".into(),
            },
            id("context"),
        ),
    ])
    .unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [
            T::standard_scale(id("p"), id("x"), 0., 0.5),
            T::identity(id("context"), id("context")),
        ],
        [FittedStage::Scalar {
            transformations: vec![
                T::positive_streak(id("x"), id("streak"), 0.),
                T::time_since_positive(id("x"), id("age"), 0.),
                T::positive_streak(id("context"), id("context_streak"), 0.),
                T::time_since_positive(id("context"), id("context_age"), 0.),
            ],
        }],
        4,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<2>::new(),
        ArrayFeatureVector::<4>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 1., 0), [1., 0.]),
        (Event::price(a, f64::MAX, 10), [f64::NAN; 2]),
        (Event::volume(a, 1., 20), [f64::NAN; 2]),
        (Event::price(a, 0., 30), [0., 30.]),
        (Event::price(a, 1., 40), [1., 0.]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(&pipeline.values()[..2], &expected);
    }
    for value in [1., 2.] {
        pipeline
            .update_context(&[("context", Some(value))])
            .unwrap();
        equal(pipeline.values(), &[1., 0., f64::NAN, f64::NAN]);
        assert_eq!(pipeline.output_observations(), &[false; 4]);
    }
    for event in [Event::price(a, 2., 39), Event::price(a, f64::NAN, 50)] {
        assert!(pipeline.handle_event(event).is_err());
        equal(pipeline.values(), &[1., 0., f64::NAN, f64::NAN]);
    }
    pipeline.handle_event(Event::volume(a, 1., 50)).unwrap();
    equal(pipeline.values(), &[1., 10., f64::NAN, f64::NAN]);
}

#[test]
fn ages_follow_monotonic_composite_clocks_after_pca() {
    let a = Symbol::new("state-pca-a").unwrap();
    let b = Symbol::new("state-pca-b").unwrap();
    let raw = FeatureExtractorSpec::new([
        raw(a).definitions()[0].clone(),
        FeatureDefinition::new(
            FeatureKey::Field {
                symbol: b,
                field: EventField::Price,
            },
            id("q"),
        ),
    ])
    .unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [T::identity(id("p"), id("p")), T::identity(id("q"), id("q"))],
        [
            FittedStage::Pca {
                outputs: vec![id("difference")],
                mean: vec![0., 0.],
                components: vec![vec![1., -1.]],
                output_scale: vec![1.],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::time_since_negative(id("difference"), id("age"), 0.),
                    T::observations_since_negative(id("difference"), id("count"), 0.),
                ],
            },
        ],
        2,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<2>::new(),
        ArrayFeatureVector::<2>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 1., 0), [f64::NAN; 2]),
        (Event::price(b, 2., 100), [0., 0.]),
        (Event::price(a, 4., 10), [0., 1.]),
        (Event::volume(a, 1., 20), [0., 1.]),
        (Event::volume(b, 1., 150), [50., 1.]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &expected);
    }
}

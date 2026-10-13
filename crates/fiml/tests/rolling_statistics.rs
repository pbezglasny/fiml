use std::time::Duration;

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, FittedStage, PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};

fn id(name: &str) -> FeatureId {
    FeatureId::new(name)
}
fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
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
            actual == expected
                || actual.is_nan() && expected.is_nan()
                || (actual - expected).abs() < 1e-12,
            "{actual} != {expected}"
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn shared_artifact_replays_rolling_statistics() {
    for text in [include_str!(
        "../../../tests/fixtures/rolling_statistics.json"
    )] {
        let fixture: serde_json::Value = serde_json::from_str(text).unwrap();
        let spec: PipelineSpec = serde_json::from_value(fixture["pipeline"].clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), fixture["pipeline"]);
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<2>::new(),
                ArrayFeatureVector::<16>::new(),
            )
            .unwrap();
        for step in fixture["steps"].as_array().unwrap() {
            let event = &step["event"];
            let timestamp = event["timestamp"].as_i64().unwrap();
            let event = match event["kind"].as_str().unwrap() {
                "time" => Event::time(timestamp),
                "volume" => Event::volume(
                    Symbol::new(event["symbol"].as_str().unwrap()).unwrap(),
                    1.,
                    timestamp,
                ),
                _ => Event::trade(
                    Symbol::new(event["symbol"].as_str().unwrap()).unwrap(),
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
}

#[test]
fn missing_intermediate_values_preserve_finite_history_and_clock_refreshes_propagate() {
    let a = Symbol::new("rolling-missing").unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw(a),
        [T::standard_scale(id("p"), id("x"), 0., 0.5)],
        [
            FittedStage::Scalar {
                transformations: vec![
                    T::rolling_std(id("x"), id("s"), 2, WarmupPolicy::FirstValue),
                    T::rolling_std_timed(id("x"), id("t"), ms(20), WarmupPolicy::FirstValue),
                ],
            },
            FittedStage::Scalar {
                transformations: vec![T::identity(id("s"), id("s")), T::identity(id("t"), id("t"))],
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
        (Event::price(a, 2., 0), [0., 0.], [true, true]),
        (Event::price(a, f64::MAX, 5), [f64::NAN; 2], [true, true]),
        (Event::volume(a, 1., 6), [f64::NAN; 2], [false, true]),
        (Event::price(a, 4., 10), [2., 2.], [true, true]),
        (Event::volume(a, 1., 30), [2., f64::NAN], [false, true]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &expected);
        assert_eq!(pipeline.output_observations(), &observed);
    }
}

#[test]
fn timed_statistics_follow_composite_clocks_and_ignore_context_refreshes() {
    let a = Symbol::new("rolling-pca-a").unwrap();
    let b = Symbol::new("rolling-pca-b").unwrap();
    let raw = FeatureExtractorSpec::new([
        FeatureDefinition::new(
            FeatureKey::Field {
                symbol: a,
                field: EventField::Price,
            },
            id("p"),
        ),
        FeatureDefinition::new(
            FeatureKey::Field {
                symbol: b,
                field: EventField::Price,
            },
            id("q"),
        ),
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
        [T::identity(id("p"), id("p")), T::identity(id("q"), id("q"))],
        [
            FittedStage::SimpleImpute {
                outputs: vec![id("p"), id("q")],
                retained_input_indices: vec![0, 1],
                replacement_values: vec![0., 0.],
                indicator_input_indices: vec![],
            },
            FittedStage::Pca {
                outputs: vec![id("sum")],
                mean: vec![0., 0.],
                components: vec![vec![1., 1.]],
                output_scale: vec![1.],
            },
            FittedStage::Scalar {
                transformations: vec![T::rolling_range_timed(
                    id("sum"),
                    id("r"),
                    ms(100),
                    WarmupPolicy::FirstValue,
                )],
            },
        ],
        1,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<3>::new(),
        ArrayFeatureVector::<1>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 2., 0), 0.),
        (Event::price(b, 10., 500), 0.),
        (Event::price(a, 4., 100), 2.), // Both 12 and 14 contribute at composite time 500.
        (Event::volume(a, 1., 200), 2.),
        (Event::volume(b, 1., 600), f64::NAN),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &[expected]);
    }
    pipeline.update_context(&[("context", Some(42.))]).unwrap();
    equal(pipeline.values(), &[f64::NAN]);
    assert_eq!(pipeline.output_observations(), &[false]);
}

#[test]
fn invalid_windows_fail_in_base_and_scalar_stages() {
    for constructor in [
        T::rolling_std,
        T::rolling_min,
        T::rolling_max,
        T::rolling_range,
    ] {
        for window in [0, usize::MAX] {
            let def = constructor(id("p"), id("s"), window, WarmupPolicy::FullWindow);
            assert!(PipelineSpec::new(raw(Symbol::GLOBAL), [def.clone()]).is_err());
            assert!(
                PipelineSpec::with_stages(
                    raw(Symbol::GLOBAL),
                    [T::identity(id("p"), id("p"))],
                    [FittedStage::Scalar {
                        transformations: vec![def]
                    }],
                    1,
                    None
                )
                .is_err()
            );
        }
    }
    for constructor in [
        T::rolling_std_timed_with_aggregation,
        T::rolling_min_timed_with_aggregation,
        T::rolling_max_timed_with_aggregation,
        T::rolling_range_timed_with_aggregation,
    ] {
        for (window, aggregation) in [
            (Duration::ZERO, ms(1)),
            (ms(10), Duration::ZERO),
            (Duration::from_nanos(1), ms(1)),
            (ms(10), ms(3)),
            (ms(1), ms(2)),
            (Duration::MAX, ms(1)),
            (ms(999999), ms(1)),
        ] {
            let def = constructor(
                id("p"),
                id("s"),
                window,
                aggregation,
                WarmupPolicy::FirstValue,
            );
            assert!(PipelineSpec::new(raw(Symbol::GLOBAL), [def.clone()]).is_err());
            assert!(
                PipelineSpec::with_stages(
                    raw(Symbol::GLOBAL),
                    [T::identity(id("p"), id("p"))],
                    [FittedStage::Scalar {
                        transformations: vec![def]
                    }],
                    1,
                    None
                )
                .is_err()
            );
        }
    }
}

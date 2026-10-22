mod common;

use std::time::Duration;

use fiml::{
    Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey, FittedStage,
    PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
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
fn shared_artifact_replays_irregular_intervals_mixed_symbols_and_event_kinds() {
    for text in [
        include_str!("../../../tests/fixtures/timed_transformers.json"),
        include_str!("../../../tests/fixtures/timed_transformers_aggregated.json"),
    ] {
        let fixture: serde_json::Value = serde_json::from_str(text).unwrap();
        let spec: PipelineSpec = serde_json::from_value(fixture["pipeline"].clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), fixture["pipeline"]);
        let mut pipeline = spec.build().unwrap();
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
            equal(pipeline.output("model").unwrap().values(), &expected);
            let observed: Vec<_> = step["observed"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            assert_eq!(pipeline.output("model").unwrap().observations(), observed);
        }
    }
}

#[test]
fn timed_refreshes_feed_later_averages_and_timed_histories_without_sampling_raw_duplicates() {
    let a = Symbol::new("timed-chained").unwrap();
    let mut pipeline = common::linear_spec(
        raw(a),
        [T::delta_timed(id("p"), id("d"), ms(100))],
        [
            FittedStage::Scalar {
                transformations: vec![
                    T::sma(id("d"), id("avg"), 2, WarmupPolicy::FullWindow),
                    T::lagged_timed(id("d"), id("lag"), ms(100)),
                ],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::identity(id("avg"), id("avg")),
                    T::identity(id("lag"), id("lag")),
                ],
            },
        ],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 2., 0), [f64::NAN, f64::NAN]),
        (Event::price(a, 4., 50), [f64::NAN, f64::NAN]),
        (Event::volume(a, 1., 100), [f64::NAN, f64::NAN]), // First finite delta = 2.
        (Event::volume(a, 1., 150), [1., f64::NAN]), // Delta = 0; raw history has no duplicate at 100.
        (Event::volume(a, 1., 200), [0., 2.]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.output("model").unwrap().values(), &expected);
        assert_eq!(
            pipeline.output("model").unwrap().observations(),
            &[true, true]
        );
    }
}

#[test]
fn clocks_follow_selection_imputation_pca_and_monotonic_composite_time() {
    let a = Symbol::new("timed-pca-a").unwrap();
    let b = Symbol::new("timed-pca-b").unwrap();
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
    let mut pipeline = common::linear_spec(
        raw,
        [
            T::identity(id("context"), id("context")),
            T::identity(id("p"), id("p")),
            T::identity(id("q"), id("q")),
        ],
        [
            FittedStage::Select {
                outputs: vec![id("p"), id("q")],
                input_indices: vec![1, 2],
            },
            FittedStage::SimpleImpute {
                outputs: vec![id("p"), id("q"), id("missing_q")],
                retained_input_indices: vec![0, 1],
                replacement_values: vec![0., 0.],
                indicator_input_indices: vec![1],
            },
            FittedStage::Pca {
                outputs: vec![id("sum")],
                mean: vec![0.; 3],
                components: vec![vec![1., 1., 0.]],
                output_scale: vec![1.],
            },
            FittedStage::Scalar {
                transformations: vec![T::delta_timed(id("sum"), id("d"), ms(100))],
            },
        ],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 2., 0), f64::NAN),
        (Event::price(b, 10., 500), 10.),
        (Event::price(a, 4., 100), 12.), // Records 14 at composite time 500, replacing 12.
        (Event::volume(a, 1., 200), 12.),
        (Event::volume(b, 1., 600), 0.),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.output("model").unwrap().values(), &[expected]);
    }
    pipeline.update_context(&[("context", Some(42.))]).unwrap();
    equal(pipeline.output("model").unwrap().values(), &[0.]);
    assert_eq!(pipeline.output("model").unwrap().observations(), &[false]);
    for event in [Event::price(a, 99., 199), Event::price(a, f64::NAN, 201)] {
        assert!(pipeline.handle_event(event).is_err());
        equal(pipeline.output("model").unwrap().values(), &[0.]);
    }
    pipeline.handle_event(Event::price(a, 8., 201)).unwrap();
    equal(pipeline.output("model").unwrap().values(), &[4.]);
}

#[test]
fn missing_scaled_inputs_recover_without_entering_history() {
    let a = Symbol::new("timed-missing").unwrap();
    let mut pipeline = common::linear_spec(
        raw(a),
        [T::standard_scale(id("p"), id("x"), 0., 0.5)],
        [FittedStage::Scalar {
            transformations: vec![
                T::lagged_timed(id("x"), id("lag"), ms(100)),
                T::delta_timed(id("x"), id("d"), ms(100)),
            ],
        }],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 2., 0), [f64::NAN; 2]),
        (Event::price(a, f64::MAX, 100), [f64::NAN; 2]),
        (Event::volume(a, 1., 200), [f64::NAN; 2]),
        (Event::price(a, 4., 300), [4., 4.]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.output("model").unwrap().values(), &expected);
    }
}

#[test]
fn validates_every_timed_kind_in_base_and_scalar_stages() {
    for constructor in [
        T::lagged_timed,
        T::delta_timed,
        T::simple_return_timed,
        T::log_return_timed,
    ] {
        for window in [Duration::ZERO, Duration::from_nanos(1), Duration::MAX] {
            let definition = constructor(id("p"), id("x"), window);
            assert!(
                common::linear_spec(raw(Symbol::GLOBAL), [definition.clone()], [], None).is_err()
            );
            assert!(
                common::linear_spec(
                    raw(Symbol::GLOBAL),
                    [T::identity(id("p"), id("p"))],
                    [FittedStage::Scalar {
                        transformations: vec![definition]
                    }],
                    None
                )
                .is_err()
            );
        }
    }
}

#[test]
fn global_calendars_and_event_count_lags_use_the_maximum_event_clock() {
    let a = Symbol::new("timed-global-a").unwrap();
    let b = Symbol::new("timed-global-b").unwrap();
    let raw_calendar = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: fiml::FeatureSource::AnyEvent,
            utc_offset_millis: 0,
        },
        id("time"),
    )])
    .unwrap();
    let mut calendar = common::linear_spec(
        raw_calendar,
        [T::lagged_timed(id("time"), id("lag"), ms(100))],
        [],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (symbol, timestamp, expected, observed) in [
        (a, 0, f64::NAN, true),
        (a, 50, f64::NAN, true),
        (b, 1000, 50., true),
        (a, 100, 50., false),
        (b, 1100, 1000., true),
    ] {
        calendar
            .handle_event(Event::volume(symbol, 1., timestamp))
            .unwrap();
        equal(calendar.output("model").unwrap().values(), &[expected]);
        assert_eq!(
            calendar.output("model").unwrap().observations(),
            &[observed]
        );
    }
    let mut arrival = common::linear_spec(
        raw(a),
        [T::lagged(id("p"), id("arrival"), 1)],
        [FittedStage::Scalar {
            transformations: vec![T::delta_timed(id("arrival"), id("d"), ms(1))],
        }],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 2., 100), f64::NAN),
        (Event::volume(b, 1., 0), f64::NAN),
        (Event::price(a, 4., 200), 0.),
        (Event::volume(b, 1., 1), 2.),
        (Event::volume(b, 1., 1000), 0.),
        (Event::price(a, 8., 201), 0.),
    ] {
        arrival.handle_event(event).unwrap();
        equal(arrival.output("model").unwrap().values(), &[expected]);
    }
}

#[test]
fn context_only_inputs_never_gain_market_observations_or_a_clock() {
    let a = Symbol::new("timed-context").unwrap();
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Context {
            symbol: a,
            name: "context".into(),
        },
        id("context"),
    )])
    .unwrap();
    let mut pipeline = common::linear_spec(
        raw,
        [T::lagged_timed(id("context"), id("lag"), ms(1))],
        [],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for timestamp in [0, 1, 1000] {
        pipeline.update_context(&[("context", Some(42.))]).unwrap();
        pipeline
            .handle_event(Event::price(a, 1., timestamp))
            .unwrap();
        equal(pipeline.output("model").unwrap().values(), &[f64::NAN]);
        assert_eq!(pipeline.output("model").unwrap().observations(), &[false]);
    }
}

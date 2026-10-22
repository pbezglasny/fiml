mod common;
use fiml::*;

fn day_of_week(id: &str) -> FeatureDefinition {
    FeatureDefinition::new(
        FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        },
        FeatureId::new(id),
    )
}

fn time_since_first_event(id: &str) -> FeatureDefinition {
    FeatureDefinition::new(
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
            utc_offset_millis: 0,
        },
        FeatureId::new(id),
    )
}

fn price(id: &str) -> FeatureDefinition {
    FeatureDefinition::new(
        FeatureKey::Field {
            symbol: Symbol::GLOBAL,
            field: EventField::Price,
        },
        FeatureId::new(id),
    )
}

#[test]
fn lagged_transformers_track_accepted_events_in_authored_output_order() {
    let raw_spec = FeatureExtractorSpec::new([time_since_first_event("elapsed")]).unwrap();
    let spec = common::linear_spec(
        raw_spec,
        [
            TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("lag2"), 2),
            TransformerDefinition::identity(FeatureId::new("elapsed"), FeatureId::new("now")),
            TransformerDefinition::standard_scale(
                FeatureId::new("elapsed"),
                FeatureId::new("scaled"),
                0.0,
                10.0,
            ),
            TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("lag1"), 1),
        ],
        [],
        None,
    )
    .unwrap();
    let mut pipeline = spec.build().unwrap();

    assert!(
        pipeline
            .output("model")
            .unwrap()
            .values()
            .iter()
            .all(|value| value.is_nan())
    );
    assert_eq!(
        pipeline.output("model").unwrap().feature_ids(),
        &[
            FeatureId::new("lag2"),
            FeatureId::new("now"),
            FeatureId::new("scaled"),
            FeatureId::new("lag1"),
        ]
    );

    let cases = [
        (100, [f64::NAN, 0.0, 0.0, f64::NAN]),
        (110, [f64::NAN, 10.0, 1.0, 0.0]),
        (120, [0.0, 20.0, 2.0, 10.0]),
        (130, [10.0, 30.0, 3.0, 20.0]),
        (140, [20.0, 40.0, 4.0, 30.0]),
    ];
    for (timestamp, expected) in cases {
        if timestamp > 100 {
            let before = pipeline
                .output("model")
                .unwrap()
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert!(pipeline.handle_event(Event::time(99)).is_err());
            assert!(
                pipeline
                    .handle_event(Event::price(Symbol::GLOBAL, f64::NAN, timestamp))
                    .is_err()
            );
            assert_eq!(pipeline.last_timestamp(), Some(timestamp - 10));
            assert_eq!(
                pipeline
                    .output("model")
                    .unwrap()
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                before
            );
        }
        pipeline.handle_event(Event::time(timestamp)).unwrap();
        assert_eq!(pipeline.raw_values(), &[(timestamp - 100) as f64]);
        for (actual, expected) in pipeline
            .output("model")
            .unwrap()
            .values()
            .iter()
            .zip(expected)
        {
            assert!((actual.is_nan() && expected.is_nan()) || *actual == expected);
        }
        assert_eq!(pipeline.output("model").unwrap().values().len(), 4);
    }
}

#[test]
fn lagged_groups_keep_distinct_inputs_and_repeated_lags_separate() {
    let raw_spec =
        FeatureExtractorSpec::new([time_since_first_event("elapsed"), day_of_week("day")]).unwrap();
    let spec = common::linear_spec(
        raw_spec,
        [
            TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("elapsed2"), 2),
            TransformerDefinition::lagged(FeatureId::new("day"), FeatureId::new("day1"), 1),
            TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("elapsed1"), 1),
            TransformerDefinition::lagged(FeatureId::new("day"), FeatureId::new("day2"), 2),
            TransformerDefinition::lagged(
                FeatureId::new("elapsed"),
                FeatureId::new("elapsed1_copy"),
                1,
            ),
        ],
        [],
        None,
    )
    .unwrap();
    let mut pipeline = spec.build().unwrap();
    let cases = [
        (100, [f64::NAN; 5]),
        (110, [f64::NAN, 4.0, 0.0, f64::NAN, 0.0]),
        (120, [0.0, 4.0, 10.0, 4.0, 10.0]),
        (130, [10.0, 4.0, 20.0, 4.0, 20.0]),
    ];
    for (timestamp, expected) in cases {
        pipeline.handle_event(Event::time(timestamp)).unwrap();
        for (actual, expected) in pipeline
            .output("model")
            .unwrap()
            .values()
            .iter()
            .zip(expected)
        {
            assert!((actual.is_nan() && expected.is_nan()) || *actual == expected);
        }
    }
}

#[test]
fn identity_and_standard_scaling_work() {
    let raw_spec = FeatureExtractorSpec::new([day_of_week("day")]).unwrap();
    let spec = common::linear_spec(
        raw_spec,
        [
            TransformerDefinition::identity(FeatureId::new("day"), FeatureId::new("identity")),
            TransformerDefinition::standard_scale(
                FeatureId::new("day"),
                FeatureId::new("scaled"),
                2.0,
                2.0,
            ),
        ],
        [],
        None,
    )
    .unwrap();
    let mut pipeline = spec.build().unwrap();

    assert!(
        pipeline
            .output("model")
            .unwrap()
            .values()
            .iter()
            .all(|value| value.is_nan())
    );
    pipeline.handle_event(Event::time(0)).unwrap();

    assert_eq!(pipeline.raw_values(), &[4.0]);
    assert_eq!(pipeline.output("model").unwrap().values(), &[4.0, 1.0]);
}

#[test]
fn authored_order_is_final_order_and_raw_and_final_ids_are_separate() {
    let raw_spec =
        FeatureExtractorSpec::new([time_since_first_event("elapsed"), day_of_week("day")]).unwrap();
    let spec = common::linear_spec(
        raw_spec,
        [
            TransformerDefinition::identity(FeatureId::new("elapsed"), FeatureId::new("elapsed")),
            TransformerDefinition::identity(FeatureId::new("day"), FeatureId::new("model_day")),
        ],
        [],
        Some("opaque".to_owned()),
    )
    .unwrap();

    assert_eq!(
        spec.build()
            .unwrap()
            .output("model")
            .unwrap()
            .values()
            .len(),
        2
    );
    assert_eq!(spec.checksum(), Some("opaque"));
    assert_eq!(spec.raw_feature_extractor_spec().feature_vector_length(), 2);
    assert_eq!(
        match &spec.nodes()[0].stage {
            FittedStage::Scalar { transformations } => transformations,
            _ => unreachable!(),
        }
        .len(),
        2
    );

    let mut pipeline = spec.build().unwrap();
    pipeline.handle_event(Event::time(0)).unwrap();

    assert_eq!(
        pipeline.output("model").unwrap().feature_ids()[0],
        FeatureId::new("elapsed")
    );
    assert_eq!(
        pipeline.output("model").unwrap().feature_ids()[1],
        FeatureId::new("model_day")
    );
    assert_eq!(pipeline.output("model").unwrap().values()[..2], [0.0, 4.0]);
    assert_eq!(pipeline.output("model").unwrap().values().len(), 2);
}

#[test]
fn undeclared_raw_features_are_omitted_and_scaling_does_not_mutate_raw_values() {
    let raw_spec =
        FeatureExtractorSpec::new([day_of_week("day"), time_since_first_event("elapsed")]).unwrap();
    let spec = common::linear_spec(
        raw_spec,
        [TransformerDefinition::standard_scale(
            FeatureId::new("day"),
            FeatureId::new("day"),
            2.0,
            2.0,
        )],
        [],
        None,
    )
    .unwrap();
    let mut pipeline = spec.build().unwrap();

    pipeline.handle_event(Event::time(0)).unwrap();

    assert_eq!(pipeline.raw_values(), &[4.0, 0.0]);
    assert_eq!(pipeline.output("model").unwrap().values(), &[1.0]);
    assert_eq!(
        pipeline.output("model").unwrap().feature_ids(),
        &[FeatureId::new("day")]
    );
}

#[test]
fn warmup_nan_propagates_and_rejected_event_leaves_final_output_unchanged() {
    let raw_spec = FeatureExtractorSpec::new([price("sma")]).unwrap();
    let spec = common::linear_spec(
        raw_spec,
        [TransformerDefinition::sma(
            FeatureId::new("sma"),
            FeatureId::new("sma"),
            2,
            WarmupPolicy::FullWindow,
        )],
        [FittedStage::Scalar {
            transformations: vec![TransformerDefinition::standard_scale(
                FeatureId::new("sma"),
                FeatureId::new("scaled_sma"),
                10.0,
                5.0,
            )],
        }],
        None,
    )
    .unwrap();
    let mut pipeline = spec.build().unwrap();

    pipeline
        .handle_event(Event::price(Symbol::GLOBAL, 10.0, 1))
        .unwrap();
    assert_eq!(pipeline.raw_values(), &[10.0]);
    assert!(pipeline.output("model").unwrap().values()[0].is_nan());

    pipeline
        .handle_event(Event::price(Symbol::GLOBAL, 20.0, 2))
        .unwrap();
    assert_eq!(pipeline.raw_values(), &[20.0]);
    assert_eq!(pipeline.output("model").unwrap().values(), &[1.0]);
    let error = match pipeline.handle_event(Event::price(Symbol::GLOBAL, 100.0, 0)) {
        Err(error) => error,
        Ok(_) => panic!("out-of-order event should fail"),
    };

    assert!(matches!(error, FimlError::TimestampOutOfOrder { .. }));
    assert_eq!(pipeline.raw_values(), &[20.0]);
    assert_eq!(pipeline.output("model").unwrap().values(), &[1.0]);
    assert_eq!(pipeline.last_timestamp(), Some(2));
}

use fiml::{
    ArrayFeatureVector, Event, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey,
    FeatureSource, FittedStage, PipelineSpec, Symbol, TransformerDefinition,
};

fn base() -> (FeatureExtractorSpec, Vec<TransformerDefinition>) {
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
            utc_offset_millis: 0,
        },
        FeatureId::new("elapsed"),
    )])
    .unwrap();
    (
        raw,
        vec![
            TransformerDefinition::identity(FeatureId::new("elapsed"), FeatureId::new("now")),
            TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("lag"), 1),
        ],
    )
}

fn stages() -> Vec<FittedStage> {
    vec![
        FittedStage::StandardScale {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            mean: vec![1.0, 2.0],
            scale: vec![2.0, 4.0],
        },
        FittedStage::Pca {
            outputs: vec![FeatureId::new("pc0")],
            mean: vec![0.5, -0.5],
            components: vec![vec![0.6, 0.8]],
            output_scale: vec![2.0],
        },
        FittedStage::StandardScale {
            outputs: vec![FeatureId::new("pc0")],
            mean: vec![1.0],
            scale: vec![2.0],
        },
    ]
}

#[test]
fn min_max_stage_preserves_nans_and_clips_only_when_configured() {
    let (raw, definitions) = base();
    let spec = PipelineSpec::with_stages(
        raw,
        definitions,
        [
            FittedStage::MinMaxScale {
                outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
                scale: vec![2.0, 4.0],
                min: vec![-1.0, 3.0],
                clip: None,
            },
            FittedStage::MinMaxScale {
                outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
                scale: vec![1.0, 1.0],
                min: vec![0.0, 0.0],
                clip: Some((-2.0, 2.0)),
            },
        ],
        2,
        None,
    )
    .unwrap();
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new(),
        )
        .unwrap();

    pipeline.handle_event(Event::time(0)).unwrap();
    assert_eq!(pipeline.values()[0], -1.0);
    assert!(pipeline.values()[1].is_nan());
    pipeline.handle_event(Event::time(2)).unwrap();
    assert_eq!(pipeline.values(), &[2.0, 2.0]);

    #[cfg(feature = "serde")]
    {
        let mut document = serde_json::to_value(&spec).unwrap();
        assert_eq!(document["version"], "2.2");
        assert_eq!(
            serde_json::from_value::<PipelineSpec>(document.clone()).unwrap(),
            spec
        );
        document["version"] = "2.1".into();
        assert!(
            serde_json::from_value::<PipelineSpec>(document)
                .unwrap_err()
                .to_string()
                .contains("MinMaxScaler stages require version 2.2")
        );
    }
}

#[test]
fn power_transform_stage_handles_limits_domains_and_overflow() {
    let (raw, mut definitions) = base();
    definitions[0] = TransformerDefinition::standard_scale(
        FeatureId::new("elapsed"),
        FeatureId::new("now"),
        1.0,
        1.0,
    );
    let spec = PipelineSpec::with_stages(
        raw,
        definitions,
        [FittedStage::PowerTransform {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            method: "yeo-johnson".into(),
            lambdas: vec![2.0, f64::EPSILON / 2.0],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, 1.0],
        }],
        2,
        None,
    )
    .unwrap();
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new(),
        )
        .unwrap();
    pipeline.handle_event(Event::time(0)).unwrap();
    assert!((pipeline.values()[0] + 2.0_f64.ln()).abs() < 1e-14);
    assert!(pipeline.values()[1].is_nan());
    pipeline.handle_event(Event::time(2)).unwrap();
    assert!((pipeline.values()[0] - 1.5).abs() < 1e-14);
    assert_eq!(pipeline.values()[1], 0.0);
    pipeline.handle_event(Event::time(4)).unwrap();
    assert!((pipeline.values()[0] - 7.5).abs() < 1e-14);
    assert!((pipeline.values()[1] - 3.0_f64.ln()).abs() < 1e-14);

    let (raw, mut definitions) = base();
    definitions[0] = TransformerDefinition::standard_scale(
        FeatureId::new("elapsed"),
        FeatureId::new("now"),
        1.0,
        1.0,
    );
    let box_cox = PipelineSpec::with_stages(
        raw,
        definitions,
        [FittedStage::PowerTransform {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            method: "box-cox".into(),
            lambdas: vec![0.0, 0.0],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, 1.0],
        }],
        2,
        None,
    )
    .unwrap();
    let mut pipeline = box_cox
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new(),
        )
        .unwrap();
    pipeline.handle_event(Event::time(0)).unwrap();
    assert!(pipeline.values().iter().all(|value| value.is_nan()));
    pipeline.handle_event(Event::time(2)).unwrap();
    assert_eq!(pipeline.values()[0], 0.0);
    assert!(pipeline.values()[1].is_nan());
    pipeline.handle_event(Event::time(4)).unwrap();
    assert!((pipeline.values()[0] - 3.0_f64.ln()).abs() < 1e-14);
    assert!((pipeline.values()[1] - 2.0_f64.ln()).abs() < 1e-14);

    let (raw, mut definitions) = base();
    definitions[0] = TransformerDefinition::standard_scale(
        FeatureId::new("elapsed"),
        FeatureId::new("now"),
        -f64::MAX,
        1.0,
    );
    let overflow = PipelineSpec::with_stages(
        raw,
        definitions,
        [FittedStage::PowerTransform {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            method: "yeo-johnson".into(),
            lambdas: vec![2.0, 1.0],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, 1.0],
        }],
        2,
        None,
    )
    .unwrap();
    let mut overflow = overflow
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new(),
        )
        .unwrap();
    overflow.handle_event(Event::time(0)).unwrap();
    assert!(overflow.values()[0].is_infinite());

    #[cfg(feature = "serde")]
    {
        let mut document = serde_json::to_value(&box_cox).unwrap();
        assert_eq!(document["version"], "2.4");
        assert_eq!(
            serde_json::from_value::<PipelineSpec>(document.clone()).unwrap(),
            box_cox
        );
        document["version"] = "2.3".into();
        assert!(
            serde_json::from_value::<PipelineSpec>(document)
                .unwrap_err()
                .to_string()
                .contains("PowerTransformer stages require version 2.4")
        );
    }
}

#[test]
fn simple_imputer_replaces_only_nans_and_appends_fitted_indicators() {
    let (raw, definitions) = base();
    let spec = PipelineSpec::with_stages(
        raw,
        definitions,
        [FittedStage::SimpleImpute {
            outputs: vec![
                FeatureId::new("now"),
                FeatureId::new("lag"),
                FeatureId::new("missingindicator_lag"),
            ],
            retained_input_indices: vec![0, 1],
            replacement_values: vec![10.0, 20.0],
            indicator_input_indices: vec![1],
        }],
        3,
        None,
    )
    .unwrap();
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<3>::new(),
        )
        .unwrap();

    pipeline.handle_event(Event::time(0)).unwrap();
    assert_eq!(pipeline.values(), &[0.0, 20.0, 1.0]);
    pipeline.handle_event(Event::time(2)).unwrap();
    assert_eq!(pipeline.values(), &[2.0, 0.0, 0.0]);

    let (raw, definitions) = base();
    let overflow = PipelineSpec::with_stages(
        raw,
        definitions,
        [
            FittedStage::Pca {
                outputs: vec![FeatureId::new("projected")],
                mean: vec![0.0, 0.0],
                components: vec![vec![f64::MAX, 0.0]],
                output_scale: vec![1.0],
            },
            FittedStage::SimpleImpute {
                outputs: vec![
                    FeatureId::new("projected"),
                    FeatureId::new("missingindicator_projected"),
                ],
                retained_input_indices: vec![0],
                replacement_values: vec![5.0],
                indicator_input_indices: vec![0],
            },
        ],
        2,
        None,
    )
    .unwrap();
    let mut overflow = overflow
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new(),
        )
        .unwrap();
    overflow.handle_event(Event::time(0)).unwrap();
    assert_eq!(overflow.values(), &[5.0, 1.0]);
    overflow.handle_event(Event::time(2)).unwrap();
    assert!(overflow.values()[0].is_infinite());
    assert_eq!(overflow.values()[1], 0.0);

    #[cfg(feature = "serde")]
    {
        let mut document = serde_json::to_value(&spec).unwrap();
        assert_eq!(document["version"], "2.3");
        assert_eq!(
            serde_json::from_value::<PipelineSpec>(document.clone()).unwrap(),
            spec
        );
        document["version"] = "2.2".into();
        assert!(
            serde_json::from_value::<PipelineSpec>(document)
                .unwrap_err()
                .to_string()
                .contains("SimpleImputer stages require version 2.3")
        );
    }
}

#[test]
fn stages_chain_after_lags_preserve_layout_warmup_and_rejection() {
    let (raw, definitions) = base();
    let spec = PipelineSpec::with_stages(raw, definitions, stages(), 2, None).unwrap();
    assert_eq!(spec.feature_vector_length(), 1);
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new_of_length(1),
        )
        .unwrap();
    pipeline.handle_event(Event::time(0)).unwrap();
    assert!(pipeline.values().iter().all(|value| value.is_nan()));
    for timestamp in [2, 4, 6] {
        pipeline.handle_event(Event::time(timestamp)).unwrap();
        let x = (timestamp as f64 - 1.0) / 2.0;
        let lag = (timestamp as f64 - 4.0) / 4.0;
        let expected = (((x - 0.5) * 0.6 + (lag + 0.5) * 0.8) / 2.0 - 1.0) / 2.0;
        assert!((pipeline.values()[0] - expected).abs() < 1e-14);
        assert!(pipeline.values()[1].is_nan());
        assert_eq!(pipeline.output_ids(), &[FeatureId::new("pc0")]);
        let before = pipeline.values()[0];
        assert!(pipeline.handle_event(Event::time(timestamp - 1)).is_err());
        assert_eq!(pipeline.values()[0], before);
    }
    // The final storage can be smaller than the two-column base and scratch.
    let (raw, definitions) = base();
    let spec = PipelineSpec::with_stages(raw, definitions, stages(), 1, None).unwrap();
    assert!(
        spec.build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<1>::new()
        )
        .is_ok()
    );
}

#[test]
fn scalar_stages_chain_after_pca_and_share_lags_without_leaking_scratch() {
    let (raw, definitions) = base();
    let mut sequence = stages();
    sequence.push(FittedStage::Scalar {
        transformations: vec![
            TransformerDefinition::lagged(FeatureId::new("pc0"), FeatureId::new("lag2"), 2),
            TransformerDefinition::identity(FeatureId::new("pc0"), FeatureId::new("now")),
            TransformerDefinition::lagged(FeatureId::new("pc0"), FeatureId::new("lag1"), 1),
            TransformerDefinition::lagged(FeatureId::new("pc0"), FeatureId::new("lag1_copy"), 1),
        ],
    });
    sequence.push(FittedStage::Scalar {
        transformations: vec![
            TransformerDefinition::standard_scale(
                FeatureId::new("lag1"),
                FeatureId::new("scaled"),
                1.0,
                2.0,
            ),
            TransformerDefinition::identity(FeatureId::new("lag2"), FeatureId::new("lag2")),
            TransformerDefinition::identity(FeatureId::new("lag1_copy"), FeatureId::new("copy")),
        ],
    });
    let spec =
        PipelineSpec::with_stages(raw.clone(), definitions.clone(), sequence, 4, None).unwrap();
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<4>::new_of_length(3),
        )
        .unwrap();
    let mut prefix = PipelineSpec::with_stages(raw, definitions, stages(), 1, None)
        .unwrap()
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<1>::new(),
        )
        .unwrap();
    let mut history = Vec::new();
    for timestamp in 0..8 {
        let event = Event::time(timestamp * 2);
        prefix.handle_event(Event::time(timestamp * 2)).unwrap();
        pipeline.handle_event(event).unwrap();
        let lag = |window| {
            history
                .len()
                .checked_sub(window)
                .map_or(f64::NAN, |index| history[index])
        };
        for (actual, expected) in
            pipeline
                .values()
                .iter()
                .zip([(lag(1) - 1.0) / 2.0, lag(2), lag(1), f64::NAN])
        {
            assert!((actual.is_nan() && expected.is_nan()) || *actual == expected);
        }
        history.push(prefix.values()[0]);
        let before = pipeline
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert!(
            pipeline
                .handle_event(Event::time(timestamp * 2 - 1))
                .is_err()
        );
        assert_eq!(
            before,
            pipeline
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        pipeline.output_ids(),
        &[
            FeatureId::new("scaled"),
            FeatureId::new("lag2"),
            FeatureId::new("copy")
        ]
    );
    #[cfg(feature = "serde")]
    {
        let document = serde_json::to_value(&spec).unwrap();
        assert_eq!(document["version"], "2.1");
        assert_eq!(
            serde_json::from_value::<PipelineSpec>(document.clone()).unwrap(),
            spec
        );
        let mut old_version = document;
        old_version["version"] = "2.0".into();
        assert!(
            serde_json::from_value::<PipelineSpec>(old_version)
                .unwrap_err()
                .to_string()
                .contains("scalar stages require version 2.1")
        );
    }
}

#[test]
fn scalar_stages_reject_inputs_outside_the_preceding_layout() {
    for definitions in [
        vec![],
        vec![TransformerDefinition::identity(
            FeatureId::new("elapsed"),
            FeatureId::new("out"),
        )],
        vec![TransformerDefinition::identity(
            FeatureId::new("future"),
            FeatureId::new("future"),
        )],
        vec![
            TransformerDefinition::identity(FeatureId::new("now"), FeatureId::new("first")),
            TransformerDefinition::identity(FeatureId::new("first"), FeatureId::new("second")),
        ],
        vec![TransformerDefinition::lagged(
            FeatureId::new("now"),
            FeatureId::new("lag"),
            0,
        )],
        vec![TransformerDefinition::standard_scale(
            FeatureId::new("now"),
            FeatureId::new("scale"),
            0.0,
            0.0,
        )],
        vec![TransformerDefinition::identity(
            FeatureId::new("now"),
            FeatureId::new("__reserved_0"),
        )],
        vec![
            TransformerDefinition::identity(FeatureId::new("now"), FeatureId::new("same")),
            TransformerDefinition::identity(FeatureId::new("lag"), FeatureId::new("same")),
        ],
    ] {
        let (raw, base) = base();
        let error = PipelineSpec::with_stages(
            raw,
            base,
            [FittedStage::Scalar {
                transformations: definitions,
            }],
            2,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("stage 0"), "{error}");
    }
}

#[test]
fn stage_numeric_and_layout_validation_is_shared_by_construction_and_json() {
    let invalid = [
        FittedStage::PowerTransform {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            method: "unknown".into(),
            lambdas: vec![1.0, 1.0],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, 1.0],
        },
        FittedStage::PowerTransform {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            method: "yeo-johnson".into(),
            lambdas: vec![f64::NAN, 1.0],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, 1.0],
        },
        FittedStage::PowerTransform {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            method: "box-cox".into(),
            lambdas: vec![1.0],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, 1.0],
        },
        FittedStage::StandardScale {
            outputs: vec![FeatureId::new("wrong")],
            mean: vec![0.0],
            scale: vec![1.0],
        },
        FittedStage::StandardScale {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            mean: vec![0.0, f64::NAN],
            scale: vec![1.0, 1.0],
        },
        FittedStage::StandardScale {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            mean: vec![0.0, 0.0],
            scale: vec![1.0, f64::from_bits(1)],
        },
        FittedStage::MinMaxScale {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            scale: vec![1.0, f64::NAN],
            min: vec![0.0, f64::NAN],
            clip: None,
        },
        FittedStage::MinMaxScale {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            scale: vec![1.0, 1.0],
            min: vec![0.0],
            clip: None,
        },
        FittedStage::MinMaxScale {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            scale: vec![1.0, 1.0],
            min: vec![0.0, 0.0],
            clip: Some((1.0, f64::INFINITY)),
        },
        FittedStage::SimpleImpute {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            retained_input_indices: vec![0, 1],
            replacement_values: vec![0.0],
            indicator_input_indices: vec![],
        },
        FittedStage::SimpleImpute {
            outputs: vec![FeatureId::new("lag"), FeatureId::new("now")],
            retained_input_indices: vec![1, 0],
            replacement_values: vec![0.0, 0.0],
            indicator_input_indices: vec![],
        },
        FittedStage::SimpleImpute {
            outputs: vec![FeatureId::new("now"), FeatureId::new("lag")],
            retained_input_indices: vec![0, 1],
            replacement_values: vec![0.0, f64::INFINITY],
            indicator_input_indices: vec![],
        },
        FittedStage::Pca {
            outputs: vec![FeatureId::new("pc")],
            mean: vec![0.0, 0.0],
            components: vec![vec![1.0]],
            output_scale: vec![1.0],
        },
        FittedStage::Pca {
            outputs: vec![FeatureId::new("pc")],
            mean: vec![0.0, 0.0],
            components: vec![vec![1.0, 0.0]],
            output_scale: vec![0.0],
        },
        FittedStage::Pca {
            outputs: vec![FeatureId::new("pc")],
            mean: vec![f64::MAX, f64::MAX],
            components: vec![vec![1.0, 1.0]],
            output_scale: vec![1.0],
        },
        FittedStage::Pca {
            outputs: vec![FeatureId::new("__reserved_0")],
            mean: vec![0.0, 0.0],
            components: vec![vec![1.0, 0.0]],
            output_scale: vec![1.0],
        },
        FittedStage::Pca {
            outputs: vec![FeatureId::new("pc"), FeatureId::new("pc")],
            mean: vec![0.0, 0.0],
            components: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            output_scale: vec![1.0, 1.0],
        },
    ];
    for stage in invalid {
        let (raw, definitions) = base();
        let error = PipelineSpec::with_stages(raw, definitions, [stage], 2, None).unwrap_err();
        assert!(error.to_string().contains("stage 0"), "{error}");
    }
}

#[cfg(feature = "serde")]
#[test]
fn json_migrates_v1_and_strictly_validates_v2_stages() {
    let (raw, definitions) = base();
    let base = PipelineSpec::new(raw, definitions).unwrap();
    let mut legacy = serde_json::to_value(&base).unwrap();
    legacy["version"] = "1.0".into();
    legacy["model_input"]
        .as_object_mut()
        .unwrap()
        .remove("stages");
    let restored: PipelineSpec = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(restored, base);
    assert_eq!(serde_json::to_value(restored).unwrap()["version"], "2.0");
    legacy["model_input"]["stages"] = serde_json::json!([]);
    assert!(serde_json::from_value::<PipelineSpec>(legacy).is_err());

    let (raw, definitions) = self::base();
    let spec = PipelineSpec::with_stages(raw, definitions, stages(), 1, None).unwrap();
    let value = serde_json::to_value(&spec).unwrap();
    assert_eq!(
        serde_json::from_value::<PipelineSpec>(value.clone()).unwrap(),
        spec
    );
    for (pointer, replacement) in [
        ("/model_input/stages", serde_json::Value::Null),
        ("/model_input/stages/1/type", serde_json::json!("unknown")),
        (
            "/model_input/stages/1/components",
            serde_json::json!([[1.0]]),
        ),
        (
            "/model_input/stages/1/output_scale",
            serde_json::json!([null]),
        ),
        ("/model_input/length", serde_json::json!(2)),
    ] {
        let mut invalid = value.clone();
        *invalid.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            serde_json::from_value::<PipelineSpec>(invalid).is_err(),
            "{pointer}"
        );
    }
    let mut missing = value.clone();
    missing["model_input"]
        .as_object_mut()
        .unwrap()
        .remove("stages");
    assert!(serde_json::from_value::<PipelineSpec>(missing).is_err());
    let mut extra = value;
    extra["model_input"]["stages"][1]["extra"] = true.into();
    assert!(serde_json::from_value::<PipelineSpec>(extra).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn standalone_rust_replays_python_fitted_sklearn_fixture() {
    use fiml::VecFeatureVector;
    use serde::Deserialize;

    /// Literal sklearn training/export output shared with the Python parity test.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        pipeline: PipelineSpec,
        symbol: String,
        timestamp: Vec<i64>,
        price: Vec<f64>,
        expected: Vec<Vec<Option<f64>>>,
    }

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/sklearn_pipeline.json"
    ))
    .unwrap();
    let raw = fixture.pipeline.raw_feature_extractor_spec();
    let mut pipeline = fixture
        .pipeline
        .build(
            VecFeatureVector::new_of_length(
                raw.feature_vector_capacity(),
                raw.feature_vector_length(),
            ),
            VecFeatureVector::new_of_length(
                fixture.pipeline.feature_vector_capacity(),
                fixture.pipeline.feature_vector_length(),
            ),
        )
        .unwrap();
    assert_eq!(fixture.timestamp.len(), fixture.price.len());
    assert_eq!(fixture.price.len(), fixture.expected.len());
    let symbol = Symbol::new(&fixture.symbol).unwrap();
    for ((timestamp, price), expected) in fixture
        .timestamp
        .into_iter()
        .zip(fixture.price)
        .zip(fixture.expected)
    {
        pipeline
            .handle_event(Event::trade(symbol, price, 1.0, timestamp, None))
            .unwrap();
        assert_eq!(pipeline.values().len(), expected.len());
        for (&actual, expected) in pipeline.values().iter().zip(expected) {
            match expected {
                Some(expected) => assert!(
                    (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                    "{actual} != {expected}"
                ),
                None => assert!(actual.is_nan()),
            }
        }
    }
}

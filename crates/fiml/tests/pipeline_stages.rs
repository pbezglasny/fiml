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
fn stage_numeric_and_layout_validation_is_shared_by_construction_and_json() {
    let invalid = [
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

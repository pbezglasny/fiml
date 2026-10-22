use fiml::{
    Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey, FittedStage,
    PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};

fn raw() -> FeatureExtractorSpec {
    FeatureExtractorSpec::with_capacity(
        [FeatureDefinition::new(
            FeatureKey::Field {
                symbol: Symbol::new("graph-test").unwrap(),
                field: EventField::Price,
            },
            FeatureId::new("price"),
        )],
        3,
    )
    .unwrap()
}

fn scalar(definitions: impl IntoIterator<Item = T>) -> FittedStage {
    FittedStage::Scalar {
        transformations: definitions.into_iter().collect(),
    }
}

#[test]
fn shared_histories_aliases_and_intermediate_outputs_match_independent_pipelines() {
    let id = FeatureId::new;
    let mut spec = PipelineSpec::new(raw());
    let root = spec.root();
    let shared = spec
        .add(
            root,
            scalar([T::sma(id("price"), id("mean"), 3, WarmupPolicy::FullWindow)]),
        )
        .unwrap();
    let left = spec
        .add(shared, scalar([T::lagged(id("mean"), id("lag"), 1)]))
        .unwrap();
    let right = spec
        .add(
            shared,
            scalar([T::standard_scale(id("mean"), id("scaled"), 1., 2.)]),
        )
        .unwrap();
    spec.output("left", left).unwrap();
    spec.output("shared", shared).unwrap();
    spec.output("alias", shared).unwrap();
    spec.output("right", right).unwrap();
    spec.output("raw", root).unwrap();
    let mut pipeline = spec.build().unwrap();
    let output_id = pipeline.output("left").unwrap().id();
    let mut independent = Vec::new();
    for stage in [
        scalar([T::lagged(id("mean"), id("lag"), 1)]),
        scalar([T::standard_scale(id("mean"), id("scaled"), 1., 2.)]),
    ] {
        let mut branch = PipelineSpec::new(raw());
        let mean = branch
            .add(
                branch.root(),
                scalar([T::sma(id("price"), id("mean"), 3, WarmupPolicy::FullWindow)]),
            )
            .unwrap();
        let last = branch.add(mean, stage).unwrap();
        branch.output("value", last).unwrap();
        independent.push(branch.build().unwrap());
    }
    assert_eq!(
        pipeline.output_names().collect::<Vec<_>>(),
        ["left", "shared", "alias", "right", "raw"]
    );
    for timestamp in 0..20 {
        let event = || {
            Event::price(
                Symbol::new("graph-test").unwrap(),
                (timestamp * timestamp + 1) as f64,
                timestamp,
            )
        };
        pipeline.handle_event(event()).unwrap();
        for (branch, name) in independent.iter_mut().zip(["left", "right"]) {
            branch.handle_event(event()).unwrap();
            assert_eq!(
                pipeline.output(name).unwrap().values()[0].to_bits(),
                branch.output("value").unwrap().values()[0].to_bits()
            );
        }
        assert_eq!(
            pipeline.output("shared").unwrap().values().as_ptr(),
            pipeline.output("alias").unwrap().values().as_ptr()
        );
        assert_eq!(
            pipeline.output_by_id(output_id).unwrap().values().as_ptr(),
            pipeline.output("left").unwrap().values().as_ptr()
        );
    }
    assert_eq!(pipeline.output("raw").unwrap().values().len(), 1);
    assert_eq!(pipeline.raw_values().len(), 3);
    let before = pipeline.output_by_id(output_id).unwrap().values()[0];
    assert!(
        pipeline
            .handle_event(Event::price(
                Symbol::new("graph-test").unwrap(),
                f64::INFINITY,
                20
            ))
            .is_err()
    );
    assert_eq!(
        pipeline.output_by_id(output_id).unwrap().values()[0],
        before
    );
    pipeline.reset().unwrap();
    assert!(pipeline.output_by_id(output_id).unwrap().values()[0].is_nan());
    assert!(spec.build().unwrap().output_by_id(output_id).is_none());
}

#[test]
fn builder_ownership_snapshots_names_and_empty_output_validation() {
    let mut spec = PipelineSpec::new(raw());
    let root = spec.root();
    let mut cloned = spec.clone();
    assert!(cloned.output("foreign", root).is_err());
    assert!(
        cloned
            .add(
                root,
                scalar([T::identity(FeatureId::new("price"), FeatureId::new("p"))])
            )
            .is_err()
    );
    assert!(spec.build().is_err());
    for name in ["", "   ", "__reserved_1"] {
        assert!(spec.output(name, root).is_err());
    }
    spec.output("raw", root).unwrap();
    let first = spec.build().unwrap();
    assert!(spec.output("raw", root).is_err());
    spec.output("copy", root).unwrap();
    assert_eq!(first.output_names().collect::<Vec<_>>(), ["raw"]);
    assert_eq!(spec.build().unwrap().output_names().len(), 2);
    let mut empty = PipelineSpec::new(FeatureExtractorSpec::new([]).unwrap());
    empty.output("empty", empty.root()).unwrap();
    assert!(
        empty
            .build()
            .err()
            .unwrap()
            .to_string()
            .contains("output \"empty\"")
    );
}

#[cfg(feature = "serde")]
#[test]
fn artifact_validates_topology_and_preserves_local_ids_order_and_fresh_ownership() {
    use serde_json::json;
    let mut spec = PipelineSpec::new(raw());
    let pca = spec
        .add(
            spec.root(),
            FittedStage::Pca {
                outputs: vec![FeatureId::new("pc0")],
                mean: vec![0.],
                components: vec![vec![1.]],
                output_scale: vec![1.],
            },
        )
        .unwrap();
    spec.output("z", pca).unwrap();
    spec.output("a", spec.root()).unwrap();
    let value = serde_json::to_value(&spec).unwrap();
    let restored: PipelineSpec = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), value);
    assert_eq!(restored, spec);
    assert!(restored.feature_ids(pca).is_err());
    for (pointer, replacement) in [
        ("/version", json!("3.0")),
        ("/nodes/1/parent", json!(1)),
        ("/nodes/1/parent", json!(100)),
        ("/nodes/1/parent", json!(-1)),
        ("/nodes/1/parent", json!(0.5)),
        ("/nodes/1/stage/mean", json!([null])),
        ("/nodes/1/stage/components", json!([[1., 2.]])),
        ("/outputs/0/node", json!(2)),
        ("/outputs/0/name", json!("a")),
        ("/nodes", json!([])),
        ("/outputs", json!([])),
    ] {
        let mut invalid = value.clone();
        *invalid.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            serde_json::from_value::<PipelineSpec>(invalid).is_err(),
            "{pointer}"
        );
    }
    for field in ["version", "nodes", "outputs"] {
        let mut invalid = value.clone();
        invalid.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<PipelineSpec>(invalid).is_err());
    }
    let mut invalid = value.clone();
    invalid["nodes"][1]["extra"] = json!(true);
    assert!(serde_json::from_value::<PipelineSpec>(invalid).is_err());
    let mut invalid = value;
    let root = invalid["nodes"][0].clone();
    invalid["nodes"].as_array_mut().unwrap().push(root);
    assert!(serde_json::from_value::<PipelineSpec>(invalid).is_err());
}

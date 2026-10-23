#![cfg(feature = "serde")]

use std::collections::BTreeMap;

use fiml::{Event, Pipeline, PipelineSpec, Symbol};
use serde::Deserialize;

/// Language-neutral accepted events shared with the Python parity follow-up.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FixtureEvent {
    Trade {
        symbol: String,
        timestamp: i64,
        price: f64,
        volume: f64,
    },
    Volume {
        symbol: String,
        timestamp: i64,
        volume: f64,
    },
    Time {
        timestamp: i64,
    },
}

impl FixtureEvent {
    fn event(&self) -> Event {
        match self {
            Self::Trade {
                symbol,
                timestamp,
                price,
                volume,
            } => Event::trade(
                Symbol::new(symbol).unwrap(),
                *price,
                *volume,
                *timestamp,
                None,
            ),
            Self::Volume {
                symbol,
                timestamp,
                volume,
            } => Event::volume(Symbol::new(symbol).unwrap(), *volume, *timestamp),
            Self::Time { timestamp } => Event::time(*timestamp),
        }
    }
}

/// Explicit layouts and hand-derived snapshots, independent of runtime execution.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedFixture {
    raw_capacity: usize,
    raw_feature_ids: Vec<String>,
    outputs: Vec<ExpectedLayout>,
    initial: ExpectedSnapshot,
    snapshots: Vec<ExpectedSnapshot>,
}

/// Authored output order and the coordinate order within each output.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedLayout {
    name: String,
    feature_ids: Vec<String>,
}

/// A completed event's outputs and independently tracked timestamp watermarks.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedSnapshot {
    last_timestamp: Option<i64>,
    symbol_timestamps: BTreeMap<String, Option<i64>>,
    raw_values: Vec<Option<f64>>,
    outputs: BTreeMap<String, ExpectedOutput>,
}

/// Processing observations remain explicit even when values are NaN or unchanged.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedOutput {
    values: Vec<Option<f64>>,
    observed: Vec<bool>,
}

fn assert_values(actual: &[f64], expected: &[Option<f64>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: width");
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        match expected {
            Some(expected) => assert!(
                (actual - expected).abs() <= 1e-12,
                "{context}[{index}]: {actual} != {expected}"
            ),
            None => assert!(
                actual.is_nan(),
                "{context}[{index}]: {actual} should be NaN"
            ),
        }
    }
}

fn assert_snapshot(
    pipeline: &Pipeline,
    fixture: &ExpectedFixture,
    snapshot: &ExpectedSnapshot,
    context: &str,
) {
    assert_eq!(
        pipeline.output_names().collect::<Vec<_>>(),
        fixture
            .outputs
            .iter()
            .map(|layout| layout.name.as_str())
            .collect::<Vec<_>>(),
        "{context}: output order"
    );
    assert_eq!(pipeline.raw_values().len(), fixture.raw_capacity);
    assert_eq!(
        pipeline
            .raw_feature_ids()
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        fixture.raw_feature_ids
    );
    assert_values(pipeline.raw_values(), &snapshot.raw_values, context);
    assert_eq!(
        pipeline.last_timestamp(),
        snapshot.last_timestamp,
        "{context}: last event"
    );
    assert_eq!(snapshot.symbol_timestamps.len(), 3);
    for symbol in ["a", "b", "__global__"] {
        assert_eq!(
            pipeline.last_timestamp_for_symbol(Symbol::new(symbol).unwrap()),
            snapshot.symbol_timestamps[symbol],
            "{context}: clock for {symbol}"
        );
    }
    assert_eq!(snapshot.outputs.len(), fixture.outputs.len());
    for (slot, layout) in fixture.outputs.iter().enumerate() {
        let output = pipeline.output(&layout.name).unwrap();
        let expected = &snapshot.outputs[&layout.name];
        let context = format!("{context}: {}", layout.name);
        assert_eq!(
            output.id(),
            pipeline.output_at(slot).unwrap().id(),
            "{context}"
        );
        assert_eq!(
            output
                .feature_ids()
                .iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>(),
            layout.feature_ids,
            "{context}: IDs"
        );
        assert_eq!(
            output.values().len(),
            layout.feature_ids.len(),
            "{context}: active width"
        );
        assert_values(output.values(), &expected.values, &context);
        assert_eq!(
            output.observations(),
            expected.observed,
            "{context}: observations"
        );
    }
    assert_eq!(
        pipeline.output("shared").unwrap().values().as_ptr(),
        pipeline.output("shared_alias").unwrap().values().as_ptr()
    );
    assert_eq!(
        pipeline.output("decision").unwrap().values().as_ptr(),
        pipeline.raw_values().as_ptr()
    );
    assert_eq!(
        pipeline.raw_observations(),
        snapshot.outputs["decision"].observed
    );
}

fn replay(
    pipeline: &mut Pipeline,
    spec: &PipelineSpec,
    events: &[FixtureEvent],
    expected: &ExpectedFixture,
    phase: &str,
) {
    let mut independent = [("model_a", 1), ("model_b", 2)].map(|(name, branch_index)| {
        let mut branch = PipelineSpec::new(spec.raw_feature_extractor_spec().clone());
        let shared = branch
            .add(branch.root(), spec.nodes()[0].stage.clone())
            .unwrap();
        let leaf = branch
            .add(shared, spec.nodes()[branch_index].stage.clone())
            .unwrap();
        branch.output(name, leaf).unwrap();
        (name, branch.build().unwrap())
    });

    assert_eq!(events.len(), expected.snapshots.len());
    assert_snapshot(pipeline, expected, &expected.initial, phase);
    for (index, (event, snapshot)) in events.iter().zip(&expected.snapshots).enumerate() {
        pipeline.handle_event(event.event()).unwrap();
        let context = format!("{phase}, event {index}");
        // Reading every output repeatedly must not advance the shared history.
        for _ in 0..2 {
            assert_snapshot(pipeline, expected, snapshot, &context);
        }
        for (name, branch) in &mut independent {
            branch.handle_event(event.event()).unwrap();
            let output = branch.output(name).unwrap();
            let expected = &snapshot.outputs[*name];
            assert_values(
                output.values(),
                &expected.values,
                &format!("{context}: independent {name}"),
            );
            assert_eq!(
                output.feature_ids(),
                pipeline.output(name).unwrap().feature_ids()
            );
            assert_eq!(
                output.observations(),
                expected.observed,
                "{context}: independent {name}"
            );
        }
    }
}

#[test]
fn rust_graph_matches_shared_multi_output_fixture() {
    let artifact =
        include_str!("../../../tests/fixtures/multiple_output_parity/pipeline_spec.json");
    let events: Vec<FixtureEvent> = serde_json::from_str(include_str!(
        "../../../tests/fixtures/multiple_output_parity/events.json"
    ))
    .unwrap();
    let expected: ExpectedFixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/multiple_output_parity/expected.json"
    ))
    .unwrap();
    let spec: PipelineSpec = serde_json::from_str(artifact).unwrap();
    assert_eq!(
        serde_json::to_value(&spec).unwrap(),
        serde_json::from_str::<serde_json::Value>(artifact).unwrap()
    );

    let mut pipeline = spec.build().unwrap();
    replay(&mut pipeline, &spec, &events, &expected, "fresh");
    pipeline.reset().unwrap();
    replay(&mut pipeline, &spec, &events, &expected, "reset");

    let restored: PipelineSpec =
        serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(restored, spec);
    replay(
        &mut restored.build().unwrap(),
        &restored,
        &events,
        &expected,
        "reloaded",
    );
}

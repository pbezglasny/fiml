#![cfg(feature = "serde")]

use std::{fs, path::PathBuf};

use fiml::{Event, ModelInputSpec, Symbol, VecFeatureVector};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FixtureEvent {
    Trade {
        symbol: String,
        timestamp: i64,
        price: f64,
        volume: f64,
    },
    Time {
        timestamp: i64,
    },
}

impl FixtureEvent {
    fn into_event(self) -> Event {
        match self {
            Self::Trade {
                symbol,
                timestamp,
                price,
                volume,
            } => Event::trade(Symbol::new(&symbol), price, volume, timestamp, None),
            Self::Time { timestamp } => Event::time(timestamp),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedFixture {
    raw: ExpectedLayout,
    model: ExpectedLayout,
    snapshots: Vec<ExpectedSnapshot>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedLayout {
    capacity: usize,
    length: usize,
    active_ids: Vec<String>,
    names: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedSnapshot {
    raw_values: Vec<Option<f64>>,
    model_values: Vec<Option<f64>>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/model_input_parity")
        .join(name)
}

fn read_fixture(name: &str) -> String {
    fs::read_to_string(fixture_path(name)).unwrap()
}

fn assert_values(actual: &[f64], expected: &[Option<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        match expected {
            Some(expected) => assert_eq!(actual, *expected, "value at index {index}"),
            None => assert!(actual.is_nan(), "value at index {index} should be NaN"),
        }
    }
}

#[test]
fn rust_pipeline_matches_shared_model_input_fixture() {
    let spec_json = read_fixture("model_input_spec.json");
    let canonical_json: Value = serde_json::from_str(&spec_json).unwrap();
    let spec: ModelInputSpec = serde_json::from_str(&spec_json).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), canonical_json);

    let events: Vec<FixtureEvent> = serde_json::from_str(&read_fixture("events.json")).unwrap();
    let expected: ExpectedFixture = serde_json::from_str(&read_fixture("expected.json")).unwrap();
    assert_eq!(events.len(), expected.snapshots.len());

    let raw_active_ids = spec
        .raw_feature_vector_spec()
        .definitions()
        .iter()
        .map(|definition| definition.id.as_str().to_owned())
        .collect::<Vec<_>>();
    let raw_vector = VecFeatureVector::new_of_length(expected.raw.capacity, expected.raw.length);
    let model_vector =
        VecFeatureVector::new_of_length(expected.model.capacity, expected.model.length);
    let mut pipeline = spec.build(raw_vector, model_vector).unwrap();

    for (event, snapshot) in events.into_iter().zip(&expected.snapshots) {
        pipeline.handle_event(event.into_event()).unwrap();

        assert_eq!(raw_active_ids, expected.raw.active_ids);
        assert_eq!(
            pipeline
                .output_ids()
                .iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>(),
            expected
                .model
                .active_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(expected.raw.names.len(), expected.raw.capacity);
        assert_eq!(expected.model.names.len(), expected.model.capacity);
        assert_eq!(raw_active_ids.len(), expected.raw.length);
        assert_eq!(pipeline.output_ids().len(), expected.model.length);
        assert_eq!(pipeline.raw_values().len(), expected.raw.capacity);
        assert_eq!(pipeline.values().len(), expected.model.capacity);
        assert_values(pipeline.raw_values(), &snapshot.raw_values);
        assert_values(pipeline.values(), &snapshot.model_values);
    }
}

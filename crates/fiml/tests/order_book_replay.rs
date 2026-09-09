#![cfg(feature = "serde")]

use fiml::order_book::{
    OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate, OrderBookSnapshot, Side,
};
use fiml::{Event, FeatureExtractorSpec, FeatureVector, PipelineSpec, Symbol, VecFeatureVector};
use rust_decimal::Decimal;
use serde_json::Value;

fn event(value: &Value) -> Event {
    let timestamp = value["timestamp"].as_i64().unwrap();
    if value["kind"] == "time" {
        return Event::time(timestamp);
    }
    let symbol = Symbol::new(value["symbol"].as_str().unwrap()).unwrap();
    if value["kind"] == "price" {
        return Event::price(symbol, value["price"].as_f64().unwrap(), timestamp);
    }
    let id = value["update_id"].as_u64().unwrap();
    let decimal = |value: &Value| Decimal::from_str_exact(value.as_str().unwrap()).unwrap();
    if value["kind"] == "snapshot" {
        let levels = |side: &str| {
            value[side]
                .as_array()
                .unwrap()
                .iter()
                .map(|level| OrderBookLevel::new(decimal(&level[0]), decimal(&level[1])))
                .collect()
        };
        Event::order_book_snapshot(
            symbol,
            timestamp,
            OrderBookSnapshot::new(id, levels("bids"), levels("asks")),
        )
    } else {
        let changes = value["changes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|change| {
                OrderBookLevelUpdate::new(
                    if change[0] == "bid" {
                        Side::Bid
                    } else {
                        Side::Ask
                    },
                    decimal(&change[1]),
                    decimal(&change[2]),
                )
            })
            .collect();
        Event::order_book_delta(symbol, timestamp, OrderBookDelta::new(id, changes))
    }
}

fn assert_values(values: &[f64], expected: &Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(values.len(), expected.len());
    for (index, (&actual, expected)) in values.iter().zip(expected).enumerate() {
        if expected.is_null() {
            assert!(actual.is_nan(), "index {index}: {actual}");
        } else {
            assert_eq!(actual, expected.as_f64().unwrap(), "index {index}");
        }
    }
}

#[test]
fn configured_raw_and_model_runtimes_match_shared_book_replay() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/order_book_replay.json"
    ))
    .unwrap();
    let spec: PipelineSpec = serde_json::from_value(fixture["pipeline"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), fixture["pipeline"]);
    let raw_spec: FeatureExtractorSpec = spec.raw_feature_extractor_spec().clone();
    let mut raw = raw_spec
        .build(VecFeatureVector::new_of_length(21, 20))
        .unwrap();
    let mut pipeline = spec
        .build(
            VecFeatureVector::new_of_length(21, 20),
            VecFeatureVector::new_of_length(5, 4),
        )
        .unwrap();
    assert!(
        raw.feature_vector()
            .values()
            .iter()
            .all(|value| value.is_nan())
    );
    assert!(pipeline.values().iter().all(|value| value.is_nan()));
    assert_eq!(
        raw.feature_ids()
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        fixture["raw_names"].as_array().unwrap()[..20]
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        pipeline
            .output_ids()
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        fixture["model_names"].as_array().unwrap()[..4]
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    for (index, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
        let previous = pipeline.last_timestamp();
        let raw_result = raw.handle_event(event(&step["event"]));
        let result = pipeline.handle_event(event(&step["event"]));
        if let Some(error) = step["error"].as_str() {
            assert!(
                result.err().unwrap().to_string().contains(error),
                "step {index}"
            );
            assert!(
                raw_result.err().unwrap().to_string().contains(error),
                "step {index}"
            );
            assert_eq!(pipeline.last_timestamp(), previous);
            assert_eq!(raw.last_timestamp(), previous);
        } else {
            result.unwrap();
            raw_result.unwrap();
        }
        assert_values(raw.feature_vector().values(), &step["raw"]);
        assert_values(pipeline.raw_values(), &step["raw"]);
        assert_values(pipeline.values(), &step["model"]);
    }
    // A spec contains configuration only: rebuilding starts with fresh state.
    let fresh = raw_spec
        .build(VecFeatureVector::new_of_length(21, 20))
        .unwrap();
    assert!(
        fresh
            .feature_vector()
            .values()
            .iter()
            .all(|value| value.is_nan())
    );
}

#[test]
fn book_configurations_validate_before_constructing_runtime_state() {
    use fiml::order_book::{OrderBookConfig, UpdatePolicy};
    use fiml::{FeatureDefinition, FeatureKey, FimlError, InvalidArgumentError};
    let btc = Symbol::new("config-btc").unwrap();
    let eth = Symbol::new("config-eth").unwrap();
    let config = OrderBookConfig::new(btc, UpdatePolicy::Contiguous, 8);
    let spec = FeatureExtractorSpec::new([FeatureDefinition::with_default_id(
        FeatureKey::OrderBookMidPrice { symbol: btc },
    )])
    .unwrap();
    assert!(matches!(
        spec.build(VecFeatureVector::new(1)),
        Err(FimlError::OrderBookNotConfigured { .. })
    ));
    assert!(matches!(
        spec.clone().with_order_books([config, config]),
        Err(FimlError::DuplicateOrderBook { .. })
    ));
    assert!(matches!(
        spec.clone().with_order_books([OrderBookConfig::new(
            Symbol::GLOBAL,
            UpdatePolicy::Monotonic,
            0
        )]),
        Err(FimlError::InvalidArgument(
            InvalidArgumentError::GlobalOrderBookSymbol
        ))
    ));
    let spec = spec
        .with_order_books([
            OrderBookConfig::new(eth, UpdatePolicy::Monotonic, 0),
            config,
        ])
        .unwrap();
    assert_eq!(spec.order_books()[0], config);
    let serialized = serde_json::to_string(&spec).unwrap();
    assert_eq!(
        serde_json::from_str::<FeatureExtractorSpec>(&serialized).unwrap(),
        spec
    );
    let raw = spec.build(VecFeatureVector::new(1)).unwrap();
    assert_eq!(
        raw.order_book_of_symbol(btc).unwrap().last_update_id(),
        None
    );
    assert_eq!(
        raw.order_book_of_symbol(eth).unwrap().last_update_id(),
        None
    );
}

#![cfg(feature = "serde")]

use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractor,
    FeatureExtractorSpec, FeatureId, FeatureKey, FeatureVector, Symbol, WarmupPolicy,
};
use serde_json::{Value, json};

#[test]
fn shared_volatility_artifact_preserves_values_warmup_ids_and_window_order() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/sample_volatility.json"
    ))
    .unwrap();
    let definitions = [
        ("eth", EventField::Price, WarmupPolicy::FullWindow, vec![1]),
        (
            "btc",
            EventField::TradePrice,
            WarmupPolicy::FullWindow,
            vec![2, 1],
        ),
        (
            "btc",
            EventField::Price,
            WarmupPolicy::FullWindow,
            vec![3, 1],
        ),
        (
            "btc",
            EventField::Price,
            WarmupPolicy::FirstValue,
            vec![3, 2],
        ),
    ]
    .into_iter()
    .flat_map(|(symbol, source, warmup_policy, windows)| {
        windows.into_iter().map(move |window| {
            FeatureDefinition::with_default_id(FeatureKey::Volatility {
                symbol: Symbol::new(symbol).unwrap(),
                source,
                window,
                warmup_policy,
            })
        })
    });
    let built_spec = FeatureExtractorSpec::new(definitions).unwrap();
    assert_eq!(serde_json::to_value(&built_spec).unwrap(), fixture["spec"]);
    assert_eq!(built_spec.indicator_count(), 4);

    for explicit_ids in [false, true] {
        let mut document = fixture["spec"].clone();
        let mut ids: Vec<_> = fixture["feature_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| FeatureId::new(id.as_str().unwrap()))
            .collect();
        if explicit_ids {
            for (index, output) in document["features"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .flat_map(|group| group["indicators"].as_array_mut().unwrap())
                .flat_map(|indicator| indicator["outputs"].as_array_mut().unwrap())
                .enumerate()
            {
                let name = format!("volatility_{index}");
                output["id"] = json!(name);
                ids[index] = FeatureId::new(name);
            }
        }
        let spec: FeatureExtractorSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
        let mut builder = FeatureExtractor::builder(ArrayFeatureVector::<7>::new());
        for definition in spec.definitions() {
            builder = builder.add_feature(definition.clone());
        }
        for mut extractor in [
            spec.build(ArrayFeatureVector::<7>::new()).unwrap(),
            builder.build().unwrap(),
        ] {
            assert_eq!(extractor.feature_ids(), ids);
            assert!(
                extractor
                    .feature_vector()
                    .values()
                    .iter()
                    .all(|v| v.is_nan())
            );
            for (step_index, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
                let event = &step["event"];
                let timestamp = event["timestamp"].as_i64().unwrap();
                let symbol = event["symbol"]
                    .as_str()
                    .map(|symbol| Symbol::new(symbol).unwrap())
                    .unwrap_or(Symbol::GLOBAL);
                let event = match event["kind"].as_str().unwrap() {
                    "price" => Event::price(symbol, event["price"].as_f64().unwrap(), timestamp),
                    "trade" => Event::trade(
                        symbol,
                        event["price"].as_f64().unwrap(),
                        event["volume"].as_f64().unwrap(),
                        timestamp,
                        None,
                    ),
                    "volume" => Event::volume(symbol, event["volume"].as_f64().unwrap(), timestamp),
                    "time" => Event::time(timestamp),
                    kind => panic!("unexpected fixture event: {kind}"),
                };
                extractor.handle_event(event).unwrap();
                for (column, (&actual, expected)) in extractor
                    .feature_vector()
                    .values()
                    .iter()
                    .zip(step["expected"].as_array().unwrap())
                    .enumerate()
                {
                    if expected.is_null() {
                        assert!(
                            actual.is_nan(),
                            "step {step_index}, column {column}: {actual}"
                        );
                    } else {
                        let expected = expected.as_f64().unwrap();
                        assert!(
                            (actual - expected).abs() < 1e-12,
                            "step {step_index}, column {column}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

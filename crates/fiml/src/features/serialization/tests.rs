use std::time::Duration;

use super::FEATURE_SET_FORMAT_VERSION;
use crate::WarmupPolicy;
use crate::features::definition::{FeatureSet, IndicatorSpec, ValueSource};

#[test]
fn serialization_emits_canonical_empty_feature_set() {
    let json = serde_json::to_string(&FeatureSet::default()).unwrap();

    assert_eq!(
        json,
        format!(r#"{{"version":"{FEATURE_SET_FORMAT_VERSION}","features":[],"options":{{}}}}"#)
    );
}

#[test]
fn checked_in_feature_set_schema_is_valid_json() {
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../../../../docs/feature-set.schema.json")).unwrap();

    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(schema["title"], "fiml FeatureSet");
}

#[test]
fn serialization_shape_covers_every_indicator_variant() {
    let feature_set = FeatureSet::builder()
        .sma_from_with_warmup(
            "BTCUSDT",
            ValueSource::Volume,
            [2, 4],
            WarmupPolicy::FullWindow,
        )
        .ema_from_with_warmup(
            "ETHUSDT",
            ValueSource::TradePrice,
            [3],
            WarmupPolicy::FirstValue,
        )
        .cvd_with_warmup("BTCUSDT", [5], WarmupPolicy::FullWindow)
        .sma_timed_from_with_warmup(
            "ETHUSDT",
            ValueSource::TradeVolume,
            Duration::from_secs(1),
            [Duration::from_secs(2)],
            WarmupPolicy::FirstValue,
        )
        .obv_timed_with_warmup(
            "BTCUSDT",
            Duration::from_millis(1),
            [Duration::from_secs(30)],
            WarmupPolicy::FullWindow,
        )
        .trade_count_timed_with_warmup(
            "BTCUSDT",
            Duration::from_millis(10),
            Duration::from_secs(60),
            WarmupPolicy::FirstValue,
        )
        .day_of_week()
        .time_since_first_event_of_day(3_600_000)
        .build();

    let expected = serde_json::json!({
        "version": FEATURE_SET_FORMAT_VERSION,
        "features": [
            {
                "indicators": [
                    {
                        "name": "day_of_week",
                        "options": {}
                    },
                    {
                        "name": "time_since_first_event_of_day",
                        "options": {
                            "utc_offset": "+01:00"
                        }
                    }
                ]
            },
            {
                "symbol": "btcusdt",
                "indicators": [
                    {
                        "name": "cvd",
                        "options": {
                            "windows": [5],
                            "warmup_policy": "full_window"
                        }
                    },
                    {
                        "name": "obv_timed",
                        "options": {
                            "aggregation": "1ms",
                            "windows": ["30s"],
                            "warmup_policy": "full_window"
                        }
                    },
                    {
                        "name": "sma",
                        "options": {
                            "source": "volume",
                            "windows": [2, 4],
                            "warmup_policy": "full_window"
                        }
                    },
                    {
                        "name": "trade_count_timed",
                        "options": {
                            "aggregation": "10ms",
                            "window": "1m",
                            "warmup_policy": "first_value"
                        }
                    }
                ]
            },
            {
                "symbol": "ethusdt",
                "indicators": [
                    {
                        "name": "ema",
                        "options": {
                            "source": "trade_price",
                            "windows": [3],
                            "warmup_policy": "first_value"
                        }
                    },
                    {
                        "name": "sma_timed",
                        "options": {
                            "source": "trade_volume",
                            "aggregation": "1s",
                            "windows": ["2s"],
                            "warmup_policy": "first_value"
                        }
                    }
                ]
            }
        ],
        "options": {}
    });

    assert_eq!(serde_json::to_value(&feature_set).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<FeatureSet>(expected).unwrap(),
        feature_set
    );
}

#[test]
fn deserialization_sorts_and_normalizes_feature_groups() {
    let json = serde_json::json!({
        "version": "1.0",
        "features": [
            {
                "symbol": "ETHUSDT",
                "indicators": [{
                    "name": "sma",
                    "options": {
                        "source": "volume",
                        "windows": [4, 2],
                        "warmup_policy": "full_window"
                    }
                }]
            },
            {
                "indicators": [{
                    "name": "day_of_week",
                    "options": {}
                }]
            },
            {
                "symbol": "btcusdt",
                "indicators": [{
                    "name": "ema",
                    "options": {
                        "source": "price",
                        "windows": [3],
                        "warmup_policy": "first_value"
                    }
                }]
            }
        ],
        "options": {}
    });

    let feature_set: FeatureSet = serde_json::from_value(json).unwrap();
    let definitions = feature_set.indicators();

    assert!(matches!(definitions[0].indicator, IndicatorSpec::DayOfWeek));
    assert_eq!(definitions[1].symbol.as_deref(), Some("btcusdt"));
    assert_eq!(definitions[2].symbol.as_deref(), Some("ethusdt"));
    match &definitions[2].indicator {
        IndicatorSpec::Sma { windows, .. } => assert_eq!(windows, &[4, 2]),
        indicator => panic!("expected SMA, got {indicator:?}"),
    }
}

#[test]
fn deserialization_rejects_duplicate_normalized_feature_groups() {
    let json = serde_json::json!({
        "version": "1.0.0",
        "features": [
            {
                "symbol": "BTCUSDT",
                "indicators": [{
                    "name": "sma",
                    "options": {
                        "source": "price",
                        "windows": [2],
                        "warmup_policy": "full_window"
                    }
                }]
            },
            {
                "symbol": "btcusdt",
                "indicators": [{
                    "name": "ema",
                    "options": {
                        "source": "price",
                        "windows": [2],
                        "warmup_policy": "full_window"
                    }
                }]
            }
        ],
        "options": {}
    });

    let error = serde_json::from_value::<FeatureSet>(json).unwrap_err();
    assert!(error.to_string().contains("duplicate feature group"));
}

#[test]
fn deserialization_rejects_invalid_group_shapes() {
    let cases = [
        (
            serde_json::json!({
                "version": "1.0.0",
                "features": [{"symbol": null, "indicators": []}],
                "options": {}
            }),
            "invalid type: null",
        ),
        (
            serde_json::json!({
                "version": "1.0.0",
                "features": [{"symbol": "btcusdt", "indicators": []}],
                "options": {}
            }),
            "must contain at least one indicator",
        ),
        (
            serde_json::json!({
                "version": "1.0.0",
                "features": [{
                    "indicators": [{
                        "name": "sma",
                        "options": {
                            "source": "price",
                            "windows": [2],
                            "warmup_policy": "full_window"
                        }
                    }]
                }],
                "options": {}
            }),
            "requires a symbol",
        ),
        (
            serde_json::json!({
                "version": "1.0.0",
                "features": [{
                    "symbol": "btcusdt",
                    "indicators": [{
                        "name": "day_of_week",
                        "options": {}
                    }]
                }],
                "options": {}
            }),
            "must omit the symbol",
        ),
    ];

    for (json, expected) in cases {
        let error = serde_json::from_value::<FeatureSet>(json).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error}"
        );
    }
}

#[test]
fn serialization_rejects_invalid_group_scope() {
    let feature_set = FeatureSet::new(vec![crate::IndicatorDef {
        symbol: Some("btcusdt".to_owned()),
        indicator: IndicatorSpec::DayOfWeek,
    }]);

    let error = serde_json::to_string(&feature_set).unwrap_err();
    assert!(error.to_string().contains("must omit the symbol"));
}

#[test]
fn deserialization_rejects_unknown_fields_and_legacy_shape() {
    let cases = [
        serde_json::json!({
            "version": "1.0.0",
            "features": [],
            "options": {"future": true}
        }),
        serde_json::json!({
            "version": "1.0.0",
            "features": [{
                "symbol": "btcusdt",
                "indicators": [{
                    "name": "sma",
                    "options": {
                        "source": "price",
                        "windows": [2],
                        "warmup_policy": "full_window",
                        "future": true
                    }
                }]
            }],
            "options": {}
        }),
        serde_json::json!({
            "version": "1.0.0",
            "indicators": []
        }),
    ];

    for json in cases {
        let error = serde_json::from_value::<FeatureSet>(json).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }
}

#[test]
fn utc_offset_input_is_canonicalized_on_output() {
    let json = r#"{
        "version": "1.0.0",
        "features": [{
            "indicators": [{
                "name": "time_since_first_event_of_day",
                "options": {"utc_offset": "UTC-5:30"}
            }]
        }],
        "options": {}
    }"#;

    let feature_set: FeatureSet = serde_json::from_str(json).unwrap();
    assert!(
        serde_json::to_string(&feature_set)
            .unwrap()
            .contains(r#""utc_offset":"-05:30""#)
    );
}

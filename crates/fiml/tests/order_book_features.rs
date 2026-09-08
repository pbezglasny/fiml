use fiml::order_book::{
    OrderBook, OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate, OrderBookSnapshot,
    OrderBookUpdate, OrderBookUpdateError, Side, UpdatePolicy,
};
use fiml::{
    ArrayFeatureVector, Event, FeatureDefinition, FeatureExtractor, FeatureExtractorSpec,
    FeatureId, FeatureKey, FeatureSource, FeatureVector, FimlError, IndicatorKind,
    InvalidIndicatorDefinitionError, Symbol,
};
use rust_decimal::{Decimal, dec};

fn keys(symbol: Symbol) -> [FeatureKey; 8] {
    [
        FeatureKey::OrderBookMidPrice { symbol },
        FeatureKey::OrderBookImbalance {
            symbol,
            n_levels: 2,
        },
        FeatureKey::OrderBookSpread { symbol },
        FeatureKey::OrderBookImbalance {
            symbol,
            n_levels: 1,
        },
        FeatureKey::OrderBookSpreadBps { symbol },
        FeatureKey::OrderBookWeightedMidPrice { symbol },
        FeatureKey::OrderBookMicroprice { symbol },
        FeatureKey::OrderBookImbalance {
            symbol,
            n_levels: 100,
        },
    ]
}

fn snapshot(id: u64) -> OrderBookSnapshot {
    OrderBookSnapshot::new(
        id,
        vec![
            OrderBookLevel::new(dec!(100), dec!(2)),
            OrderBookLevel::new(dec!(99), dec!(6)),
        ],
        vec![
            OrderBookLevel::new(dec!(102), dec!(3)),
            OrderBookLevel::new(dec!(103), dec!(1)),
        ],
    )
}

fn delta(symbol: Symbol, timestamp: i64, id: u64, size: Decimal) -> Event {
    Event::order_book_delta(
        symbol,
        timestamp,
        OrderBookDelta::new(
            id,
            vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), size)],
        ),
    )
}

fn assert_values(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}",
        );
    }
}

#[test]
fn derives_all_indicators_with_grouped_depths_and_missing_values() {
    let symbol = Symbol::new("book-values").unwrap();
    let definitions = keys(symbol).map(FeatureDefinition::with_default_id);
    let mut book = OrderBook::new(UpdatePolicy::Contiguous, 8);
    book.apply_update(OrderBookUpdate::Snapshot(snapshot(0)))
        .unwrap();
    let mut builder =
        FeatureExtractor::builder(ArrayFeatureVector::<8>::new()).add_order_book(symbol, book);
    for definition in &definitions {
        builder = builder.add_feature(definition.clone());
    }
    let mut extractor = builder.build().unwrap();
    assert!(
        extractor
            .feature_vector()
            .values()
            .iter()
            .all(|v| v.is_nan())
    );
    assert_eq!(
        extractor.feature_ids(),
        [0, 1, 3, 7, 2, 4, 5, 6].map(|i| definitions[i].id.clone())
    );

    let result = extractor
        .handle_event(Event::order_book_snapshot(symbol, 1, snapshot(1)))
        .unwrap();
    assert_eq!(result.features_updated, 6);
    assert_values(
        extractor.feature_vector().values(),
        &[
            101.0,
            1.0 / 3.0,
            -0.2,
            1.0 / 3.0,
            2.0,
            20_000.0 / 101.0,
            101.2,
            100.8,
        ],
    );

    assert_eq!(
        extractor
            .handle_event(delta(symbol, 2, 2, dec!(4)))
            .unwrap()
            .features_updated,
        6
    );
    assert_values(
        extractor.feature_vector().values(),
        &[
            101.0,
            3.0 / 7.0,
            1.0 / 7.0,
            3.0 / 7.0,
            2.0,
            20_000.0 / 101.0,
            706.0 / 7.0,
            708.0 / 7.0,
        ],
    );

    // Missing sides clear earlier values; imbalance still uses the available side.
    for (id, bids, asks, imbalance) in [
        (
            3,
            vec![OrderBookLevel::new(dec!(100), dec!(2))],
            vec![],
            1.0,
        ),
        (
            4,
            vec![],
            vec![OrderBookLevel::new(dec!(102), dec!(3))],
            -1.0,
        ),
        (5, vec![], vec![], f64::NAN),
    ] {
        extractor
            .handle_event(Event::order_book_snapshot(
                symbol,
                id as i64,
                OrderBookSnapshot::new(id, bids, asks),
            ))
            .unwrap();
        assert_values(
            extractor.feature_vector().values(),
            &[
                f64::NAN,
                imbalance,
                imbalance,
                imbalance,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
        );
    }
    extractor
        .handle_event(Event::order_book_snapshot(
            symbol,
            6,
            OrderBookSnapshot::new(
                6,
                vec![OrderBookLevel::new(dec!(0), dec!(2))],
                vec![OrderBookLevel::new(dec!(0), dec!(3))],
            ),
        ))
        .unwrap();
    assert_values(
        extractor.feature_vector().values(),
        &[0.0, -0.2, -0.2, -0.2, 0.0, f64::NAN, 0.0, 0.0],
    );
}

#[test]
fn updates_only_visible_book_state_and_preserves_rejected_event_atomicity() {
    let symbol = Symbol::new("book-routing").unwrap();
    let other = Symbol::new("other-book").unwrap();
    let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<4>::new())
        .add_order_book(symbol, OrderBook::new(UpdatePolicy::Contiguous, 8))
        .add_order_book(other, OrderBook::new(UpdatePolicy::Monotonic, 8))
        .add_feature(FeatureDefinition::new(
            FeatureKey::OrderBookImbalance {
                symbol,
                n_levels: 1,
            },
            FeatureId::new("imbalance"),
        ))
        .add_feature(FeatureDefinition::with_default_id(
            FeatureKey::OrderBookMidPrice { symbol: other },
        ))
        .add_feature(FeatureDefinition::with_default_id(FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        }))
        .add_feature(FeatureDefinition::with_default_id(FeatureKey::DayOfWeek {
            symbol,
            source: FeatureSource::Event(fiml::EventKind::OrderBookDelta),
        }))
        .build()
        .unwrap();
    assert_eq!(extractor.feature_ids()[0].as_str(), "imbalance");

    // A pre-snapshot delta is buffered, then replayed before derivation.
    assert_eq!(
        extractor
            .handle_event(delta(symbol, 0, 2, dec!(4)))
            .unwrap()
            .features_updated,
        2
    );
    assert!(extractor.feature_vector().values()[0].is_nan());
    extractor
        .handle_event(Event::order_book_snapshot(symbol, 1, snapshot(1)))
        .unwrap();
    assert_values(
        extractor.feature_vector().values(),
        &[1.0 / 7.0, f64::NAN, 4.0, 4.0],
    );

    for event in [
        delta(symbol, 2, 2, dec!(20)),
        Event::price(symbol, 999.0, 3),
        Event::time(4),
    ] {
        extractor.handle_event(event).unwrap();
        assert_values(
            extractor.feature_vector().values(),
            &[1.0 / 7.0, f64::NAN, 4.0, 4.0],
        );
    }
    extractor
        .handle_event(Event::order_book_snapshot(other, 5, snapshot(1)))
        .unwrap();
    let before = [1.0 / 7.0, 101.0, 4.0, 4.0];
    assert_values(extractor.feature_vector().values(), &before);

    assert!(matches!(
        extractor.handle_event(delta(symbol, 86_400_000, 3, dec!(-1))),
        Err(FimlError::OrderBookUpdateError {
            reason: OrderBookUpdateError::InvalidUpdate { .. }
        })
    ));
    assert_values(extractor.feature_vector().values(), &before);
    assert_eq!(extractor.last_timestamp(), Some(5));
    assert!(matches!(
        extractor.handle_event(delta(symbol, 86_400_000, 4, dec!(9))),
        Err(FimlError::OrderBookUpdateError {
            reason: OrderBookUpdateError::SequenceGap {
                expected: 3,
                received: 4
            }
        })
    ));
    assert_values(extractor.feature_vector().values(), &before);
    assert_eq!(extractor.last_timestamp(), Some(5));

    // Later buffered changes remain invisible until resynchronization.
    extractor
        .handle_event(delta(symbol, 6, 5, dec!(12)))
        .unwrap();
    assert_values(extractor.feature_vector().values(), &before);
    assert_eq!(
        extractor
            .handle_event(Event::order_book_snapshot(symbol, 7, snapshot(3)))
            .unwrap()
            .features_updated,
        2
    );
    assert_values(extractor.feature_vector().values(), &[0.6, 101.0, 4.0, 4.0]);
    assert_eq!(
        extractor
            .order_book_of_symbol(symbol)
            .unwrap()
            .last_update_id(),
        Some(5)
    );
}

#[test]
fn validates_book_requirements_definitions_and_group_limit() {
    let symbol = Symbol::new("book-validation").unwrap();
    for key in keys(symbol) {
        let definition = FeatureDefinition::with_default_id(key);
        assert!(
            matches!(FeatureExtractor::builder(ArrayFeatureVector::<1>::new()).add_feature(definition.clone()).build(), Err(FimlError::OrderBookNotConfigured { symbol: missing }) if missing == symbol)
        );
        assert!(
            matches!(FeatureExtractorSpec::new([definition]).unwrap().build(ArrayFeatureVector::<1>::new()), Err(FimlError::OrderBookNotConfigured { symbol: missing }) if missing == symbol)
        );
    }
    let key = FeatureKey::OrderBookImbalance {
        symbol,
        n_levels: 0,
    };
    assert!(matches!(
        FeatureExtractor::builder(ArrayFeatureVector::<1>::new())
            .add_feature(FeatureDefinition::with_default_id(key))
            .build(),
        Err(FimlError::InvalidIndicatorDefinition {
            indicator: IndicatorKind::OrderBookImbalance,
            reason: InvalidIndicatorDefinitionError::OrderBookDepthZero,
            ..
        })
    ));

    for (second_key, second_id, expected) in [
        (
            FeatureKey::OrderBookMidPrice { symbol },
            "second",
            InvalidIndicatorDefinitionError::DuplicateFeatureKey,
        ),
        (
            FeatureKey::OrderBookSpread { symbol },
            "first",
            InvalidIndicatorDefinitionError::DuplicateFeatureId,
        ),
    ] {
        let result = FeatureExtractor::builder(ArrayFeatureVector::<2>::new())
            .add_feature(FeatureDefinition::new(
                FeatureKey::OrderBookMidPrice { symbol },
                FeatureId::new("first"),
            ))
            .add_feature(FeatureDefinition::new(
                second_key,
                FeatureId::new(second_id),
            ))
            .build();
        assert!(
            matches!(result, Err(FimlError::InvalidIndicatorDefinition { reason, .. }) if reason == expected)
        );
    }
    for length in [16, 17] {
        let mut builder =
            FeatureExtractor::builder(ArrayFeatureVector::<17>::new_of_length(length))
                .add_order_book(symbol, OrderBook::new(UpdatePolicy::Monotonic, 1));
        for n_levels in 1..=length {
            builder = builder.add_feature(FeatureDefinition::with_default_id(
                FeatureKey::OrderBookImbalance { symbol, n_levels },
            ));
        }
        let result = builder.build();
        if length == 16 {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result,
                Err(FimlError::InvalidIndicatorDefinition {
                    reason: InvalidIndicatorDefinitionError::CompatibleGroupOutputLimitExceeded {
                        limit: 16
                    },
                    ..
                })
            ));
        }
    }
}

#[test]
fn book_ids_and_spec_grouping_are_stable() {
    let symbol = Symbol::new("BTCUSD").unwrap();
    let definitions = keys(symbol).map(FeatureDefinition::with_default_id);
    for (definition, (kind, suffix)) in definitions.iter().zip([
        ("mid_price", ""),
        ("imbalance", ":n_levels=2"),
        ("spread", ""),
        ("imbalance", ":n_levels=1"),
        ("spread_bps", ""),
        ("weighted_mid_price", ""),
        ("microprice", ""),
        ("imbalance", ":n_levels=100"),
    ]) {
        assert_eq!(
            definition.id.as_str(),
            format!("order_book_{kind}:symbol=6:btcusd:source=order_book{suffix}")
        );
    }
    let spec = FeatureExtractorSpec::new(definitions.clone()).unwrap();
    assert_eq!(spec.indicator_count(), 6);
    assert_eq!(
        spec.definitions()
            .iter()
            .map(|d| d.id.clone())
            .collect::<Vec<_>>(),
        [0, 2, 4, 5, 6, 1, 3, 7].map(|i| definitions[i].id.clone())
    );
}

#[cfg(feature = "serde")]
#[test]
fn book_definitions_round_trip_with_grouped_depths_and_custom_ids() {
    let symbol = Symbol::new("book-json").unwrap();
    let mut definitions = keys(symbol).map(FeatureDefinition::with_default_id);
    definitions[1].id = FeatureId::new("imbalance_two");
    let spec = FeatureExtractorSpec::new(definitions).unwrap();
    let json = serde_json::to_value(&spec).unwrap();
    let indicators = json["features"][0]["indicators"].as_array().unwrap();
    assert_eq!(indicators.len(), 6);
    for indicator in indicators {
        assert_eq!(
            indicator["source"],
            serde_json::json!({"type": "order_book"})
        );
        assert!(indicator.get("warmup_policy").is_none());
        assert!(indicator.get("options").is_none());
    }
    assert!(indicators[0].get("outputs").is_none());
    assert_eq!(
        indicators[5]["outputs"],
        serde_json::json!([
            {"n_levels": 2, "id": "imbalance_two"}, {"n_levels": 1}, {"n_levels": 100},
        ])
    );
    let restored: FeatureExtractorSpec = serde_json::from_value(json).unwrap();
    assert_eq!(restored, spec);
    let mut builder = FeatureExtractor::builder(ArrayFeatureVector::<8>::new())
        .add_order_book(symbol, OrderBook::new(UpdatePolicy::Contiguous, 1));
    for definition in restored.definitions() {
        builder = builder.add_feature(definition.clone());
    }
    let mut extractor = builder.build().unwrap();
    extractor
        .handle_event(Event::order_book_snapshot(symbol, 0, snapshot(0)))
        .unwrap();
    assert_values(
        extractor.feature_vector().values(),
        &[
            101.0,
            2.0,
            20_000.0 / 101.0,
            101.2,
            100.8,
            1.0 / 3.0,
            -0.2,
            1.0 / 3.0,
        ],
    );
}

#[cfg(feature = "serde")]
#[test]
fn book_json_rejects_inapplicable_parameters_and_invalid_depths() {
    use serde_json::json;

    let base = json!({
        "kind": "order_book_imbalance", "source": {"type": "order_book"},
        "outputs": [{"n_levels": 1}],
    });
    for invalid in [
        json!({"outputs": []}),
        json!({"outputs": [{}]}),
        json!({"outputs": [{"n_levels": 0}]}),
        json!({"outputs": [{"n_levels": -1}]}),
        json!({"outputs": [{"n_levels": 1.5}]}),
        json!({"outputs": [{"n_levels": "1"}]}),
        json!({"outputs": [{"n_levels": null}]}),
        json!({"outputs": [{"n_levels": 1, "window": 1}]}),
        json!({"outputs": (1..=17).map(|n| json!({"n_levels": n})).collect::<Vec<_>>()}),
        json!({"kind": "order_book_mid_price"}),
        json!({"kind": "order_book_mid_price", "outputs": [{}, {}]}),
        json!({"source": {"type": "order_book", "event": "order_book_delta"}}),
        json!({"source": {"type": "order_book", "field": "price"}}),
        json!({"source": {"type": "event", "event": "order_book_snapshot"}}),
        json!({"source": {"type": "field", "event": "trade", "field": "price"}}),
        json!({"kind": "day_of_week"}),
        json!({"warmup_policy": "full_window"}),
        json!({"options": {"aggregation": "1s"}}),
        json!({"options": {"n_levels": 1}}),
    ] {
        let mut indicator = base.clone();
        indicator
            .as_object_mut()
            .unwrap()
            .extend(invalid.as_object().unwrap().clone());
        let length = indicator["outputs"].as_array().map_or(1, Vec::len);
        let document = json!({"version": "1.0", "capacity": length, "length": length,
            "features": [{"symbol": "btcusd", "indicators": [indicator]}]});
        assert!(
            serde_json::from_value::<FeatureExtractorSpec>(document.clone()).is_err(),
            "accepted {document}"
        );
    }

    // An event indicator cannot silently accept the new output parameter.
    let document = json!({"version": "1.0", "capacity": 1, "length": 1,
    "features": [{"symbol": "btcusd", "indicators": [{
        "kind": "sma", "source": {"type": "field", "event": "trade", "field": "price"},
        "warmup_policy": "full_window", "outputs": [{"window": 5, "n_levels": 1}]
    }]}]});
    assert!(serde_json::from_value::<FeatureExtractorSpec>(document).is_err());

    let symbol = Symbol::new("book-json-invalid").unwrap();
    for depths in [vec![0], (1..=17).collect()] {
        let spec = FeatureExtractorSpec::new(depths.into_iter().map(|n_levels| {
            FeatureDefinition::with_default_id(FeatureKey::OrderBookImbalance { symbol, n_levels })
        }))
        .unwrap();
        assert!(serde_json::to_value(spec).is_err());
    }
    let mut global = json!({"version": "1.0", "capacity": 1, "length": 1,
        "features": [{"symbol": "__global__", "indicators": [base]}]});
    assert!(serde_json::from_value::<FeatureExtractorSpec>(global.clone()).is_err());
    global["features"][0]["symbol"] = json!("btcusd");
    global["features"][0]["indicators"][0]
        .as_object_mut()
        .unwrap()
        .remove("outputs");
    assert!(serde_json::from_value::<FeatureExtractorSpec>(global).is_err());
}

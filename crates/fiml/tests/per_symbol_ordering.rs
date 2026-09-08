use std::time::Duration;

use fiml::{
    ArrayFeatureVector, Event, EventField, EventKind, FeatureDefinition, FeatureExtractor,
    FeatureExtractorSpec, FeatureId, FeatureKey, FeatureSource, FeatureVector, FimlError,
    PipelineSpec, Symbol, TransformerDefinition, WarmupPolicy,
    order_book::{OrderBookDelta, OrderBookSnapshot},
};

#[test]
fn ordering_tracks_all_symbols_and_event_kinds_without_advancing_on_errors() {
    let btc = Symbol::new("ordering-btc").unwrap();
    let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<0>::new())
        .build()
        .unwrap();
    let eth = Symbol::new("ordering-created-after-construction").unwrap();
    assert_eq!(extractor.last_timestamp_for_symbol(eth), None);
    for event in [
        Event::price(btc, 1.0, 100),
        Event::trade(eth, 1.0, 1.0, -100, None),
        Event::volume(btc, 1.0, 101),
        Event::price(eth, 1.0, -100),
        Event::time(i64::MIN),
    ] {
        let symbol = event.symbol();
        let timestamp = event.timestamp();
        assert_eq!(extractor.handle_event(event).unwrap().features_updated, 0);
        assert_eq!(extractor.last_timestamp(), Some(timestamp));
        assert_eq!(extractor.last_timestamp_for_symbol(symbol), Some(timestamp));
    }
    for event in [
        Event::price(btc, f64::NAN, 99), // Ordering precedes numeric validation.
        Event::volume(btc, 1.0, 99),
        Event::trade(btc, 1.0, 1.0, 99, None),
        Event::order_book_delta(btc, 99, OrderBookDelta::new(0, Vec::new())),
        Event::order_book_snapshot(btc, 99, OrderBookSnapshot::new(0, Vec::new(), Vec::new())),
        Event::volume(eth, 1.0, -101),
    ] {
        let symbol = event.symbol();
        let event_kind = event.kind();
        let timestamp = event.timestamp();
        let previous_timestamp = extractor.last_timestamp_for_symbol(symbol).unwrap();
        assert!(
            matches!(extractor.handle_event(event), Err(FimlError::TimestampOutOfOrder {
            symbol: actual_symbol, event_kind: actual_kind, timestamp: actual_timestamp,
            previous_timestamp: actual_previous,
        }) if (actual_symbol, actual_kind, actual_timestamp, actual_previous)
            == (symbol, event_kind, timestamp, previous_timestamp))
        );
        assert_eq!(
            extractor.last_timestamp_for_symbol(symbol),
            Some(previous_timestamp)
        );
        assert_eq!(extractor.last_timestamp(), Some(i64::MIN));
    }
    assert!(
        extractor
            .handle_event(Event::price(eth, f64::INFINITY, 1_000))
            .is_err()
    );
    assert_eq!(extractor.last_timestamp_for_symbol(eth), Some(-100));
    extractor.handle_event(Event::price(eth, 1.0, -99)).unwrap();
    extractor
        .handle_event(Event::price(Symbol::GLOBAL, 1.0, -1))
        .unwrap();
    assert!(matches!(
        extractor.handle_event(Event::time(-2)),
        Err(FimlError::TimestampOutOfOrder {
            previous_timestamp: -1,
            ..
        })
    ));
}

fn timed_spec(symbols: &[Symbol], warmup_policy: WarmupPolicy) -> FeatureExtractorSpec {
    FeatureExtractorSpec::new(symbols.iter().flat_map(|&symbol| {
        let aggregation = Duration::from_millis(10);
        let window = Duration::from_millis(20);
        [
            FeatureKey::SmaTimed {
                symbol,
                source: FeatureSource::Field(EventField::TradePrice),
                aggregation,
                window,
                warmup_policy,
            },
            FeatureKey::ObvTimed {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                aggregation,
                window,
                warmup_policy,
            },
            FeatureKey::TradeCountTimed {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                aggregation,
                window,
                warmup_policy,
            },
        ]
        .map(FeatureDefinition::with_default_id)
    }))
    .unwrap()
}

fn timed_event(symbol: Symbol, kind: EventKind, timestamp: i64, price: f64) -> Event {
    match kind {
        EventKind::Trade => Event::trade(symbol, price, 2.0, timestamp, None),
        EventKind::Price => Event::price(symbol, price, timestamp),
        EventKind::Volume => Event::volume(symbol, 1.0, timestamp),
        EventKind::OrderBookDelta => {
            Event::order_book_delta(symbol, timestamp, OrderBookDelta::new(0, Vec::new()))
        }
        EventKind::OrderBookSnapshot => Event::order_book_snapshot(
            symbol,
            timestamp,
            OrderBookSnapshot::new(0, Vec::new(), Vec::new()),
        ),
        EventKind::Time => Event::time(timestamp),
    }
}

#[test]
fn timed_features_match_independent_symbol_replay_through_warmup_and_expiration() {
    let btc = Symbol::new("timed-ordering-btc").unwrap();
    let eth = Symbol::new("timed-ordering-eth").unwrap();
    for policy in [WarmupPolicy::FirstValue, WarmupPolicy::FullWindow] {
        let mut combined = timed_spec(&[btc, eth], policy)
            .build(ArrayFeatureVector::<6>::new())
            .unwrap();
        let mut independent = [btc, eth].map(|symbol| {
            timed_spec(&[symbol], policy)
                .build(ArrayFeatureVector::<3>::new())
                .unwrap()
        });
        for (symbol, kind, timestamp, price) in [
            (btc, EventKind::Trade, -20, 10.0),
            (eth, EventKind::Trade, -100, 20.0),
            (btc, EventKind::Trade, -10, 11.0),
            (eth, EventKind::Volume, -90, 0.0),
            (Symbol::GLOBAL, EventKind::Time, 100_000, 0.0),
            (btc, EventKind::Price, 0, 0.0),
            (eth, EventKind::Trade, -80, 19.0),
            (btc, EventKind::OrderBookDelta, 10, 0.0),
            (eth, EventKind::OrderBookSnapshot, -60, 0.0),
            (btc, EventKind::Trade, 40, 12.0),
            (eth, EventKind::Trade, -50, 21.0),
            (btc, EventKind::Volume, 100, 0.0),
        ] {
            let update = combined
                .handle_event(timed_event(symbol, kind, timestamp, price))
                .unwrap();
            assert_eq!(
                update.features_updated,
                if symbol == Symbol::GLOBAL { 0 } else { 3 }
            );
            if let Some(index) = [btc, eth].iter().position(|&candidate| candidate == symbol) {
                independent[index]
                    .handle_event(timed_event(symbol, kind, timestamp, price))
                    .unwrap();
            }
            for reference in &independent {
                for (id, &expected) in reference
                    .feature_ids()
                    .iter()
                    .zip(reference.feature_vector().values())
                {
                    let actual =
                        combined.feature_vector().values()[combined.feature_index(id).unwrap()];
                    assert!(
                        actual == expected || (actual.is_nan() && expected.is_nan()),
                        "{id:?}: {actual} != {expected}"
                    );
                }
            }
        }
        // Same-symbol non-trade events expired every count; both policies warmed up.
        for symbol in [btc, eth] {
            let id = FeatureId::from(&FeatureKey::TradeCountTimed {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                aggregation: Duration::from_millis(10),
                window: Duration::from_millis(20),
                warmup_policy: policy,
            });
            let value = combined.feature_vector().values()[combined.feature_index(&id).unwrap()];
            assert_eq!(value, if symbol == btc { 0.0 } else { 1.0 });
        }
    }
}

#[test]
fn global_calendars_follow_maximum_time_while_scoped_calendars_follow_their_routes() {
    let btc = Symbol::new("calendar-ordering-btc").unwrap();
    let eth = Symbol::new("calendar-ordering-eth").unwrap();
    let definitions = [
        FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        },
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
            utc_offset_millis: 0,
        },
        FeatureKey::DayOfWeek {
            symbol: eth,
            source: FeatureSource::Event(EventKind::Trade),
        },
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol: eth,
            source: FeatureSource::AnyEvent,
            utc_offset_millis: 0,
        },
    ];
    let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<4>::new());
    for key in definitions {
        extractor = extractor.add_feature(FeatureDefinition::with_default_id(key));
    }
    let mut extractor = extractor.build().unwrap();
    for (event, expected) in [
        (Event::price(btc, 1.0, -10), [3.0, 0.0, f64::NAN, f64::NAN]),
        (Event::price(btc, 1.0, 10), [4.0, 0.0, f64::NAN, f64::NAN]),
        (Event::price(btc, 1.0, 20), [4.0, 10.0, f64::NAN, f64::NAN]),
        (
            Event::trade(eth, 1.0, 1.0, -20, None),
            [4.0, 10.0, 3.0, 0.0],
        ),
        (Event::time(-100), [4.0, 10.0, 3.0, 0.0]),
        (Event::volume(eth, 1.0, -10), [4.0, 10.0, 3.0, 10.0]),
        (Event::volume(eth, 1.0, 20), [4.0, 10.0, 3.0, 0.0]),
        (Event::trade(eth, 1.0, 1.0, 25, None), [4.0, 15.0, 4.0, 5.0]),
    ] {
        extractor.handle_event(event).unwrap();
        for (&actual, expected) in extractor.feature_vector().values().iter().zip(expected) {
            assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
        }
    }
}

#[test]
fn pipeline_lags_follow_arrival_order_and_expose_symbol_timestamps() {
    let btc = Symbol::new("pipeline-ordering-btc").unwrap();
    let eth = Symbol::new("pipeline-ordering-eth").unwrap();
    let spec = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Sma {
            symbol: btc,
            source: FeatureSource::Field(EventField::Price),
            window: 1,
            warmup_policy: WarmupPolicy::FirstValue,
        },
        FeatureId::new("price"),
    )])
    .unwrap();
    let mut pipeline = PipelineSpec::new(
        spec,
        [TransformerDefinition::lagged(
            FeatureId::new("price"),
            FeatureId::new("lag"),
            1,
        )],
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<1>::new(),
    )
    .unwrap();
    assert_eq!(pipeline.last_timestamp_for_symbol(btc), None);
    pipeline.handle_event(Event::price(btc, 10.0, 100)).unwrap();
    assert!(pipeline.values()[0].is_nan());
    pipeline.handle_event(Event::price(btc, 11.0, 101)).unwrap();
    assert_eq!(pipeline.values(), &[10.0]);
    pipeline.handle_event(Event::price(eth, 20.0, 90)).unwrap();
    assert_eq!(pipeline.values(), &[11.0]);
    assert_eq!(pipeline.last_timestamp(), Some(90));
    assert_eq!(pipeline.last_timestamp_for_symbol(btc), Some(101));
    assert_eq!(pipeline.last_timestamp_for_symbol(eth), Some(90));
    assert!(pipeline.handle_event(Event::volume(btc, 1.0, 99)).is_err());
    assert_eq!(pipeline.values(), &[11.0]);
    assert_eq!(pipeline.last_timestamp(), Some(90));
}

use std::time::Duration;

use fiml::{
    ArrayFeatureVector, Event, EventField, EventKind, FeatureDefinition, FeatureExtractorSpec,
    FeatureId, FeatureKey, FeatureSource, FimlError, FittedStage, PipelineSpec, ReturnKind,
    SourceKind, SourceObservation, Symbol, TradeSide, TransformerDefinition as T, VecFeatureVector,
    WarmupPolicy,
    order_book::{
        OrderBook, OrderBookConfig, OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate,
        OrderBookSnapshot, OrderBookSyncState, OrderBookUpdate, OrderBookUpdateError,
        OrderBookUpdateOutcome as Outcome, Side, UpdatePolicy,
    },
};
use rust_decimal::dec;

fn id(name: &str) -> FeatureId {
    FeatureId::new(name)
}

fn feature(name: &str, key: FeatureKey) -> FeatureDefinition {
    FeatureDefinition::new(key, id(name))
}

fn snapshot(sequence: u64) -> OrderBookSnapshot {
    OrderBookSnapshot::new(
        sequence,
        vec![OrderBookLevel::new(dec!(100), dec!(2))],
        vec![OrderBookLevel::new(dec!(102), dec!(3))],
    )
}

fn delta(sequence: u64) -> OrderBookDelta {
    OrderBookDelta::new(
        sequence,
        vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(2))],
    )
}

#[test]
fn raw_identity_sources_and_masks_follow_active_grouped_layouts() {
    let symbol = Symbol::new("metadata-layout").unwrap();
    let returns = |lag| FeatureKey::Return {
        symbol,
        source: FeatureSource::Field(EventField::Price),
        kind: ReturnKind::Simple,
        lag,
    };
    let raw = FeatureExtractorSpec::with_capacity(
        [
            feature("return1", returns(1)),
            feature(
                "position",
                FeatureKey::Context {
                    symbol,
                    name: "position".into(),
                },
            ),
            feature("return2", returns(2)),
            feature(
                "price",
                FeatureKey::Field {
                    symbol,
                    field: EventField::Price,
                },
            ),
        ],
        6,
    )
    .unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [
            T::lagged(id("price"), id("lag"), 1),
            T::identity(id("price"), id("now")),
        ],
        [FittedStage::StandardScale {
            outputs: vec![id("lag"), id("now")],
            mean: vec![0.0; 2],
            scale: vec![1.0; 2],
        }],
        3,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<6>::new_of_length(4),
        ArrayFeatureVector::<3>::new_of_length(2),
    )
    .unwrap();

    assert_eq!(
        pipeline.raw_feature_ids(),
        [id("price"), id("position"), id("return1"), id("return2")]
    );
    assert_eq!(pipeline.output_ids(), [id("lag"), id("now")]);
    for (index, name) in ["price", "position", "return1", "return2"]
        .iter()
        .enumerate()
    {
        assert_eq!(pipeline.raw_feature_index(&id(name)), Some(index));
    }
    assert_eq!(pipeline.raw_feature_index(&id("__reserved_0")), None);
    assert_eq!(pipeline.raw_feature_index(&id("missing")), None);
    assert_eq!(pipeline.raw_feature_sources(4), None);
    assert_eq!(pipeline.raw_feature_sources(usize::MAX), None);
    assert_eq!(pipeline.raw_feature_sources(1), Some([].as_slice()));
    assert_eq!(
        pipeline.raw_feature_sources(0).unwrap(),
        &[SourceObservation {
            symbol,
            kind: SourceKind::Event(EventKind::Price),
            timestamp_millis: None,
        }]
    );
    assert_eq!(pipeline.raw_observations(), [false; 4]);
    assert_eq!(pipeline.output_observations(), [false; 2]);

    pipeline
        .handle_event(Event::price(symbol, 10.0, 100))
        .unwrap();
    assert_eq!(pipeline.raw_observations(), [true, false, true, true]);
    assert_eq!(pipeline.output_observations(), [true, true]);
    assert!(pipeline.raw_values()[2].is_nan());
    assert!(pipeline.values()[0].is_nan());
    for index in [0, 2, 3] {
        assert_eq!(
            pipeline.raw_feature_sources(index).unwrap()[0].timestamp_millis,
            Some(100)
        );
    }

    pipeline.update_context(&[("position", Some(2.0))]).unwrap();
    assert_eq!(pipeline.raw_observations(), [false; 4]);
    assert_eq!(pipeline.output_observations(), [false; 2]);
    assert!(pipeline.values()[0].is_nan()); // Context did not advance lag history.
    assert_eq!(pipeline.raw_feature_sources(1), Some([].as_slice()));
    pipeline.handle_event(Event::time(200)).unwrap();
    assert_eq!(pipeline.raw_observations(), [false; 4]);
    assert_eq!(pipeline.output_observations(), [true, false]);
    assert_eq!(pipeline.values()[..2], [10.0, 10.0]);
    assert_eq!(
        pipeline.raw_feature_sources(0).unwrap()[0].timestamp_millis,
        Some(100)
    );
    assert!(
        pipeline.raw_values()[4..]
            .iter()
            .all(|value| value.is_nan())
    );
    assert!(pipeline.values()[2].is_nan());

    // An accepted, numerically unchanged observation refreshes its source.
    pipeline
        .handle_event(Event::price(symbol, 10.0, 201))
        .unwrap();
    assert_eq!(
        pipeline.raw_feature_sources(0).unwrap()[0].timestamp_millis,
        Some(201)
    );
    assert!(
        pipeline
            .handle_event(Event::price(symbol, f64::NAN, 202))
            .is_err()
    );
    assert!(
        pipeline
            .handle_event(Event::price(symbol, 20.0, 200))
            .is_err()
    );
    assert_eq!(
        pipeline.raw_feature_sources(0).unwrap()[0].timestamp_millis,
        Some(201)
    );
    pipeline.handle_event(Event::time(203)).unwrap();
    assert_eq!(pipeline.raw_observations(), [false; 4]);
}

#[test]
fn timed_recalculation_does_not_refresh_market_inputs() {
    let symbol = Symbol::new("metadata-timed").unwrap();
    let other = Symbol::new("metadata-timed-other").unwrap();
    let aggregation = Duration::from_millis(1000);
    let window = Duration::from_millis(2000);
    let warmup_policy = WarmupPolicy::FirstValue;
    let source = FeatureSource::Event(EventKind::Trade);
    let keys = [
        FeatureKey::SmaTimed {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            aggregation,
            window,
            warmup_policy,
        },
        FeatureKey::VolatilityTimed {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            aggregation,
            window,
            warmup_policy,
        },
        FeatureKey::ObvTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        },
        FeatureKey::TradeCountTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        },
        FeatureKey::TradeVolumeTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        },
        FeatureKey::VwapTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        },
    ];
    let price_ids = [FeatureId::from(&keys[0]), FeatureId::from(&keys[1])];
    let raw = FeatureExtractorSpec::new(keys.map(FeatureDefinition::with_default_id)).unwrap();
    let mut pipeline = PipelineSpec::new(raw, [])
        .unwrap()
        .build(
            ArrayFeatureVector::<6>::new(),
            ArrayFeatureVector::<0>::new(),
        )
        .unwrap();
    pipeline
        .handle_event(Event::price(symbol, 100.0, 0))
        .unwrap();
    for index in 0..6 {
        if price_ids.contains(&pipeline.raw_feature_ids()[index]) {
            continue;
        }
        assert_eq!(
            pipeline.raw_feature_sources(index).unwrap()[0].timestamp_millis,
            None
        );
    }
    pipeline
        .handle_event(Event::trade(symbol, 100.0, 1.0, 1, None))
        .unwrap();
    pipeline
        .handle_event(Event::volume(symbol, 10.0, 2000))
        .unwrap();
    assert!(pipeline.raw_observations().iter().all(|observed| *observed));
    for index in 0..6 {
        assert_eq!(
            pipeline.raw_feature_sources(index).unwrap()[0].timestamp_millis,
            Some(if price_ids.contains(&pipeline.raw_feature_ids()[index]) {
                0
            } else {
                1
            })
        );
    }
    pipeline
        .handle_event(Event::price(symbol, 100.0, 3000))
        .unwrap();
    pipeline
        .handle_event(Event::trade(other, 100.0, 1.0, 4000, None))
        .unwrap();
    pipeline.handle_event(Event::time(5000)).unwrap();
    for index in 0..6 {
        let metadata = pipeline.raw_feature_sources(index).unwrap()[0];
        assert_eq!(metadata.symbol, symbol);
        assert_eq!(
            metadata.kind,
            SourceKind::Event(if price_ids.contains(&pipeline.raw_feature_ids()[index]) {
                EventKind::Price
            } else {
                EventKind::Trade
            })
        );
        assert_eq!(
            metadata.timestamp_millis,
            Some(if price_ids.contains(&pipeline.raw_feature_ids()[index]) {
                3000
            } else {
                1
            })
        );
    }
}

#[test]
fn source_acceptance_respects_trade_side_and_calendar_routing() {
    let btc = Symbol::new("metadata-btc").unwrap();
    let eth = Symbol::new("metadata-eth").unwrap();
    let mut extractor = FeatureExtractorSpec::new([
        feature(
            "cvd",
            FeatureKey::Cvd {
                symbol: btc,
                source: FeatureSource::Event(EventKind::Trade),
                window: 2,
                warmup_policy: WarmupPolicy::FullWindow,
            },
        ),
        feature(
            "volume",
            FeatureKey::Field {
                symbol: btc,
                field: EventField::TradeVolume,
            },
        ),
        feature(
            "global",
            FeatureKey::DayOfWeek {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::AnyEvent,
            },
        ),
        feature(
            "eth",
            FeatureKey::DayOfWeek {
                symbol: eth,
                source: FeatureSource::AnyEvent,
            },
        ),
    ])
    .unwrap()
    .build(ArrayFeatureVector::<4>::new())
    .unwrap();
    let [cvd, volume, global, eth_clock] =
        ["cvd", "volume", "global", "eth"].map(|name| extractor.feature_index(&id(name)).unwrap());
    extractor
        .handle_event(Event::trade(btc, 100.0, 1.0, 100, None))
        .unwrap();
    assert_eq!(
        extractor.feature_sources(cvd).unwrap()[0].timestamp_millis,
        None
    );
    assert_eq!(
        extractor.feature_sources(volume).unwrap()[0].timestamp_millis,
        Some(100)
    );
    extractor
        .handle_event(Event::trade(
            btc,
            100.0,
            1.0,
            101,
            Some(TradeSide::AggressorBuy),
        ))
        .unwrap();
    extractor
        .handle_event(Event::trade(btc, 100.0, 1.0, 102, None))
        .unwrap();
    assert_eq!(
        extractor.feature_sources(cvd).unwrap()[0].timestamp_millis,
        Some(101)
    );
    assert_eq!(
        extractor.feature_sources(volume).unwrap()[0].timestamp_millis,
        Some(102)
    );
    // Cross-symbol timestamp order must not rewind the global clock input.
    extractor.handle_event(Event::price(eth, 10.0, 50)).unwrap();
    assert_eq!(
        extractor.feature_sources(global).unwrap()[0].timestamp_millis,
        Some(102)
    );
    assert_eq!(
        extractor.feature_sources(eth_clock).unwrap()[0].timestamp_millis,
        Some(50)
    );
    extractor.handle_event(Event::time(200)).unwrap();
    assert_eq!(
        extractor.feature_sources(global).unwrap()[0].kind,
        SourceKind::AnyEvent
    );
    assert_eq!(
        extractor.feature_sources(global).unwrap()[0].timestamp_millis,
        Some(200)
    );
    assert_eq!(
        extractor.feature_sources(eth_clock).unwrap()[0].timestamp_millis,
        Some(50)
    );
}

#[test]
fn book_outcomes_and_replay_times_work_with_or_without_subscribers() {
    let symbol = Symbol::new("metadata-book").unwrap();
    let other = Symbol::new("metadata-book-other").unwrap();
    for subscribed in [false, true] {
        let count = usize::from(subscribed);
        let raw = FeatureExtractorSpec::new(
            subscribed.then(|| feature("bid", FeatureKey::OrderBookBestBidPrice { symbol })),
        )
        .unwrap()
        .with_order_books([OrderBookConfig::new(symbol, UpdatePolicy::Contiguous, 8)])
        .unwrap();
        let mut pipeline = PipelineSpec::new(raw, [])
            .unwrap()
            .build(VecFeatureVector::new(count), ArrayFeatureVector::<0>::new())
            .unwrap();
        let check = |pipeline: &fiml::Pipeline<VecFeatureVector, ArrayFeatureVector<0>>,
                     state,
                     timestamp| {
            let book = pipeline.order_book_of_symbol(symbol).unwrap();
            assert_eq!(book.sync_state(), state);
            assert_eq!(book.source_timestamp(), timestamp);
            if subscribed {
                assert_eq!(
                    pipeline.raw_feature_sources(0).unwrap(),
                    &[SourceObservation {
                        symbol,
                        kind: SourceKind::OrderBook,
                        timestamp_millis: timestamp,
                    }]
                );
            }
        };
        check(&pipeline, OrderBookSyncState::AwaitingSnapshot, None);
        let result = pipeline
            .handle_event(Event::order_book_delta(symbol, 150, delta(11)))
            .unwrap();
        assert_eq!(result.order_book_outcome, Some(Outcome::Buffered));
        assert_eq!(result.features_updated, 0);
        check(&pipeline, OrderBookSyncState::AwaitingSnapshot, None);
        let result = pipeline
            .handle_event(Event::order_book_snapshot(symbol, 200, snapshot(10)))
            .unwrap();
        assert_eq!(result.order_book_outcome, Some(Outcome::Applied));
        assert_eq!(result.features_updated, count);
        assert_eq!(pipeline.raw_observations(), vec![true; count]);
        check(&pipeline, OrderBookSyncState::Live, Some(150));
        assert_eq!(pipeline.last_timestamp_for_symbol(symbol), Some(200));
        let result = pipeline
            .handle_event(Event::order_book_delta(symbol, 201, delta(11)))
            .unwrap();
        assert_eq!(result.order_book_outcome, Some(Outcome::IgnoredStale));
        assert_eq!(result.features_updated, 0);
        assert_eq!(pipeline.raw_observations(), vec![false; count]);
        check(&pipeline, OrderBookSyncState::Live, Some(150));

        let result = pipeline
            .handle_event(Event::order_book_delta(symbol, 202, delta(12)))
            .unwrap();
        assert_eq!(result.order_book_outcome, Some(Outcome::Applied));
        assert_eq!(result.features_updated, count); // Levels are numerically unchanged.
        check(&pipeline, OrderBookSyncState::Live, Some(202));
        let result = pipeline
            .handle_event(Event::trade(symbol, 100.0, 1.0, 203, None))
            .unwrap();
        assert_eq!(result.order_book_outcome, None);
        check(&pipeline, OrderBookSyncState::Live, Some(202));
        assert_eq!(
            pipeline
                .handle_event(Event::order_book_snapshot(other, 204, snapshot(0)))
                .unwrap()
                .order_book_outcome,
            None
        );

        assert!(matches!(
            pipeline.handle_event(Event::order_book_delta(symbol, 250, delta(14))),
            Err(FimlError::OrderBookUpdateError {
                reason: OrderBookUpdateError::SequenceGap { .. }
            })
        ));
        check(&pipeline, OrderBookSyncState::RequireResync, Some(202));
        assert_eq!(
            pipeline
                .order_book_of_symbol(symbol)
                .unwrap()
                .best_bid()
                .unwrap()
                .price,
            dec!(100)
        );
        assert_eq!(pipeline.last_timestamp_for_symbol(symbol), Some(203));
        // A snapshot with missing history is rejected without publishing a source time.
        assert!(
            pipeline
                .handle_event(Event::order_book_snapshot(symbol, 300, snapshot(11)))
                .is_err()
        );
        check(&pipeline, OrderBookSyncState::RequireResync, Some(202));
        let result = pipeline
            .handle_event(Event::order_book_delta(symbol, 260, delta(15)))
            .unwrap();
        assert_eq!(result.order_book_outcome, Some(Outcome::Buffered));
        check(&pipeline, OrderBookSyncState::RequireResync, Some(202));
        let result = pipeline
            .handle_event(Event::order_book_snapshot(symbol, 300, snapshot(13)))
            .unwrap();
        assert_eq!(result.order_book_outcome, Some(Outcome::Resynchronized));
        assert_eq!(result.features_updated, count);
        check(&pipeline, OrderBookSyncState::Live, Some(260));
        assert_eq!(
            pipeline
                .order_book_of_symbol(symbol)
                .unwrap()
                .last_update_id(),
            Some(15)
        );
        let invalid = OrderBookDelta::new(
            16,
            vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(-1))],
        );
        assert!(
            pipeline
                .handle_event(Event::order_book_delta(symbol, 301, invalid))
                .is_err()
        );
        check(&pipeline, OrderBookSyncState::Live, Some(260));
        pipeline
            .handle_event(Event::order_book_snapshot(symbol, 302, snapshot(16)))
            .unwrap();
        check(&pipeline, OrderBookSyncState::Live, Some(302));
    }
}

#[test]
fn standalone_books_and_feature_sources_preserve_supplied_clock_values() {
    let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);
    book.apply_update(OrderBookUpdate::Delta(delta(1)), -10)
        .unwrap();
    book.apply_update(OrderBookUpdate::Snapshot(snapshot(0)), 20)
        .unwrap();
    assert_eq!(book.source_timestamp(), Some(-10));
    book.apply_update(OrderBookUpdate::Delta(delta(2)), i64::MAX)
        .unwrap();
    assert_eq!(book.source_timestamp(), Some(i64::MAX));
    assert!(
        book.apply_update(OrderBookUpdate::Delta(delta(4)), 30)
            .is_err()
    );
    assert_eq!(book.source_timestamp(), Some(i64::MAX));
    assert_eq!(
        book.apply_update(OrderBookUpdate::Snapshot(snapshot(3)), 40)
            .unwrap(),
        Outcome::Resynchronized
    );
    assert_eq!(book.source_timestamp(), Some(30)); // Original time of a retained rejected delta.

    let symbol = Symbol::new("metadata-clock").unwrap();
    let mut extractor = FeatureExtractorSpec::new([feature(
        "price",
        FeatureKey::Field {
            symbol,
            field: EventField::Price,
        },
    )])
    .unwrap()
    .build(ArrayFeatureVector::<1>::new())
    .unwrap();
    for timestamp in [-10, i64::MAX] {
        extractor
            .handle_event(Event::price(symbol, 1.0, timestamp))
            .unwrap();
        assert_eq!(
            extractor.feature_sources(0).unwrap()[0].timestamp_millis,
            Some(timestamp)
        );
    }
}

#[test]
fn continuing_one_symbol_does_not_refresh_another_live_book() {
    let btc = Symbol::new("source-books-btc").unwrap();
    let eth = Symbol::new("source-books-eth").unwrap();
    let raw = FeatureExtractorSpec::new([btc, eth].map(|symbol| {
        FeatureDefinition::with_default_id(FeatureKey::OrderBookBestBidPrice { symbol })
    }))
    .unwrap()
    .with_order_books(
        [btc, eth].map(|symbol| OrderBookConfig::new(symbol, UpdatePolicy::Contiguous, 4)),
    )
    .unwrap();
    let mut pipeline = PipelineSpec::new(raw, [])
        .unwrap()
        .build(
            ArrayFeatureVector::<2>::new(),
            ArrayFeatureVector::<0>::new(),
        )
        .unwrap();
    let eth_index = pipeline
        .raw_feature_index(&FeatureId::from(&FeatureKey::OrderBookBestBidPrice {
            symbol: eth,
        }))
        .unwrap();
    for symbol in [btc, eth] {
        pipeline
            .handle_event(Event::order_book_snapshot(symbol, 100, snapshot(0)))
            .unwrap();
    }
    pipeline
        .handle_event(Event::order_book_delta(btc, 10_000, delta(1)))
        .unwrap();
    pipeline
        .handle_event(Event::trade(eth, 100.0, 1.0, 10_000, None))
        .unwrap();
    let eth_book = pipeline.order_book_of_symbol(eth).unwrap();
    assert_eq!(eth_book.sync_state(), OrderBookSyncState::Live);
    assert!(pipeline.raw_values()[eth_index].is_finite());
    assert_eq!(pipeline.last_timestamp_for_symbol(eth), Some(10_000));
    assert_eq!(eth_book.source_timestamp(), Some(100));
    assert_eq!(
        pipeline.raw_feature_sources(eth_index).unwrap()[0].timestamp_millis,
        Some(100)
    );
    pipeline
        .handle_event(Event::order_book_delta(eth, 10_001, delta(1)))
        .unwrap();
    assert_eq!(
        pipeline.raw_feature_sources(eth_index).unwrap()[0].timestamp_millis,
        Some(10_001)
    );
}

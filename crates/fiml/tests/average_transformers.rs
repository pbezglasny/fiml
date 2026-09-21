use fiml::HeapRingBuffer;
use fiml::indicators::{ExponentialMovingAverage, SimpleMovingAverage};
use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, FittedStage, PipelineSpec, Symbol, TransformerDefinition as T, VecFeatureVector,
    WarmupPolicy,
};

fn id(name: &str) -> FeatureId {
    FeatureId::new(name)
}
fn field(symbol: Symbol, name: &str, field: EventField) -> FeatureDefinition {
    FeatureDefinition::new(FeatureKey::Field { symbol, field }, id(name))
}
fn equal(actual: &[f64], expected: &[f64]) {
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-10,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn averages_match_calculators_and_sample_only_matching_finite_observations() {
    let symbol = Symbol::new("average-parity").unwrap();
    let other = Symbol::new("average-other").unwrap();
    for warmup in [WarmupPolicy::FullWindow, WarmupPolicy::FirstValue] {
        let raw =
            FeatureExtractorSpec::new([field(symbol, "price", EventField::TradePrice)]).unwrap();
        let mut pipeline = PipelineSpec::new(
            raw,
            [
                T::sma(id("price"), id("s3"), 3, warmup),
                T::ema(id("price"), id("e1"), 1, warmup),
                T::sma(id("price"), id("s1"), 1, warmup),
                T::ema(id("price"), id("e3"), 3, warmup),
                T::sma(id("price"), id("s3-copy"), 3, warmup),
            ],
        )
        .unwrap()
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<5>::new(),
        )
        .unwrap();
        let mut sma = SimpleMovingAverage::<HeapRingBuffer<f64>, 2>::new_heap(3, warmup);
        let mut ema = ExponentialMovingAverage::<2>::new(warmup);
        for window in [3, 1] {
            sma.add_window(window).unwrap();
            ema.add_window(window).unwrap();
        }
        assert!(pipeline.values().iter().all(|v| v.is_nan()));
        for (timestamp, value) in [10., 10., 20., -2., 0., 50.].into_iter().enumerate() {
            let timestamp = timestamp as i64;
            pipeline
                .handle_event(Event::trade(symbol, value, 1., timestamp, None))
                .unwrap();
            sma.update(value);
            ema.update(value);
            let expected = [
                sma.value_at(0),
                ema.value_at(1),
                sma.value_at(1),
                ema.value_at(0),
                sma.value_at(0),
            ]
            .map(|v| v.unwrap_or(f64::NAN));
            equal(pipeline.values(), &expected);
            for event in [
                Event::trade(other, 99., 1., timestamp, None),
                Event::volume(symbol, 99., timestamp),
                Event::time(timestamp),
            ] {
                pipeline.handle_event(event).unwrap();
                equal(pipeline.values(), &expected);
            }
            assert!(
                pipeline
                    .handle_event(Event::trade(symbol, f64::NAN, 1., timestamp, None))
                    .is_err()
            );
            assert!(
                pipeline
                    .handle_event(Event::price(symbol, 1., timestamp - 1))
                    .is_err()
            );
            equal(pipeline.values(), &expected);
        }
    }
}

#[test]
fn fields_route_all_scalar_sources_and_start_missing() {
    let symbol = Symbol::new("all-fields").unwrap();
    let raw = FeatureExtractorSpec::new([
        field(symbol, "p", EventField::Price),
        field(symbol, "v", EventField::Volume),
        field(symbol, "tp", EventField::TradePrice),
        field(symbol, "tv", EventField::TradeVolume),
    ])
    .unwrap();
    let mut pipeline = PipelineSpec::new(
        raw,
        ["p", "v", "tp", "tv"].map(|name| T::identity(id(name), id(name))),
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<4>::new(),
        ArrayFeatureVector::<4>::new(),
    )
    .unwrap();
    assert!(pipeline.values().iter().all(|v| v.is_nan()));
    pipeline.handle_event(Event::price(symbol, 7., 0)).unwrap();
    equal(pipeline.values(), &[7., f64::NAN, f64::NAN, f64::NAN]);
    pipeline.handle_event(Event::volume(symbol, 8., 1)).unwrap();
    pipeline
        .handle_event(Event::trade(symbol, 9., 10., 2, None))
        .unwrap();
    equal(pipeline.values(), &[7., 8., 9., 10.]);
}

#[test]
fn successive_stages_retain_visible_outputs_in_reused_scratch() {
    let symbol = Symbol::new("average-stages").unwrap();
    let raw = FeatureExtractorSpec::new([field(symbol, "p", EventField::Price)]).unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [T::standard_scale(id("p"), id("x"), 10., 2.)],
        [
            FittedStage::Scalar {
                transformations: vec![T::sma(id("x"), id("x"), 2, WarmupPolicy::FullWindow)],
            },
            FittedStage::Scalar {
                transformations: vec![T::ema(id("x"), id("x"), 2, WarmupPolicy::FirstValue)],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::lagged(id("x"), id("lag"), 1),
                    T::identity(id("x"), id("now")),
                ],
            },
        ],
        2,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<2>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(symbol, 10., 0), [f64::NAN, f64::NAN]),
        (Event::price(symbol, 14., 1), [f64::NAN, 1.]),
        (Event::volume(symbol, 99., 2), [1., 1.]),
        (Event::price(symbol, 18., 3), [1., 7. / 3.]),
        (Event::time(4), [7. / 3., 7. / 3.]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &expected);
    }
}

#[test]
fn book_missing_outputs_preserve_history_and_ignore_buffered_updates() {
    use fiml::order_book::{
        OrderBookConfig, OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate, OrderBookSnapshot,
        Side, UpdatePolicy,
    };
    use rust_decimal::dec;
    let symbol = Symbol::new("average-book").unwrap();
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::OrderBookImbalance {
            symbol,
            n_levels: 1,
        },
        id("book"),
    )])
    .unwrap()
    .with_order_books([OrderBookConfig::new(symbol, UpdatePolicy::Contiguous, 4)])
    .unwrap();
    let mut pipeline = PipelineSpec::new(
        raw,
        [
            T::sma(id("book"), id("s"), 2, WarmupPolicy::FullWindow),
            T::ema(id("book"), id("e"), 2, WarmupPolicy::FirstValue),
        ],
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<2>::new(),
    )
    .unwrap();
    let snapshot = |seq, ts, bid, ask| {
        Event::order_book_snapshot(
            symbol,
            ts,
            OrderBookSnapshot::new(
                seq,
                vec![OrderBookLevel::new(dec!(10), bid)],
                vec![OrderBookLevel::new(dec!(11), ask)],
            ),
        )
    };
    pipeline
        .handle_event(snapshot(10, 0, dec!(3), dec!(1)))
        .unwrap();
    equal(pipeline.values(), &[f64::NAN, 0.5]);
    assert!(
        pipeline
            .handle_event(Event::order_book_delta(
                symbol,
                1,
                OrderBookDelta::new(
                    12,
                    vec![OrderBookLevelUpdate::new(Side::Bid, dec!(10), dec!(1))]
                )
            ))
            .is_err()
    );
    pipeline
        .handle_event(Event::order_book_delta(
            symbol,
            1,
            OrderBookDelta::new(12, vec![]),
        ))
        .unwrap();
    equal(pipeline.values(), &[f64::NAN, 0.5]);
    pipeline
        .handle_event(snapshot(11, 2, dec!(3), dec!(1)))
        .unwrap();
    equal(pipeline.values(), &[0.25, 1. / 6.]);
    pipeline
        .handle_event(Event::order_book_snapshot(
            symbol,
            3,
            OrderBookSnapshot::new(13, vec![], vec![]),
        ))
        .unwrap();
    equal(pipeline.values(), &[f64::NAN, f64::NAN]);
    pipeline.handle_event(Event::time(4)).unwrap();
    equal(pipeline.values(), &[f64::NAN, f64::NAN]);
    pipeline
        .handle_event(snapshot(14, 5, dec!(3), dec!(1)))
        .unwrap();
    equal(pipeline.values(), &[0.25, 7. / 18.]);
}

#[test]
fn pca_selection_and_imputation_propagate_source_observations() {
    let a = Symbol::new("async-a").unwrap();
    let b = Symbol::new("async-b").unwrap();
    let raw = FeatureExtractorSpec::new([
        field(a, "a", EventField::Price),
        field(b, "b", EventField::Price),
    ])
    .unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [T::identity(id("a"), id("a")), T::identity(id("b"), id("b"))],
        [
            FittedStage::SimpleImpute {
                outputs: vec![id("a"), id("b"), id("missing_b")],
                retained_input_indices: vec![0, 1],
                replacement_values: vec![0., 0.],
                indicator_input_indices: vec![1],
            },
            FittedStage::Select {
                outputs: vec![id("a"), id("b")],
                input_indices: vec![0, 1],
            },
            FittedStage::Pca {
                outputs: vec![id("sum")],
                mean: vec![0., 0.],
                components: vec![vec![1., 1.]],
                output_scale: vec![1.],
            },
            FittedStage::Scalar {
                transformations: vec![T::sma(id("sum"), id("avg"), 2, WarmupPolicy::FullWindow)],
            },
        ],
        1,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<2>::new(),
        ArrayFeatureVector::<1>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 10., 0), f64::NAN),
        (Event::time(1), f64::NAN),
        (Event::price(b, 20., 2), 20.),
        (Event::price(a, 10., 3), 30.),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &[expected]);
    }
}

#[test]
fn validates_average_windows_without_lag_cap_and_limits_compatible_groups() {
    let raw = FeatureExtractorSpec::new([field(Symbol::GLOBAL, "p", EventField::Price)]).unwrap();
    assert!(
        PipelineSpec::new(
            raw.clone(),
            [T::sma(id("p"), id("s"), 0, WarmupPolicy::FullWindow)]
        )
        .is_err()
    );
    let spec = PipelineSpec::new(
        raw.clone(),
        [T::sma(id("p"), id("s"), 10_001, WarmupPolicy::FullWindow)],
    )
    .unwrap();
    spec.build(VecFeatureVector::new(1), VecFeatureVector::new(1))
        .unwrap();
    assert!(
        PipelineSpec::new(
            raw,
            (0..17).map(|i| T::ema(
                id("p"),
                id(&format!("e{i}")),
                i + 1,
                WarmupPolicy::FullWindow
            ))
        )
        .is_err()
    );
}

#[test]
fn overflow_is_missing_and_lags_before_averages_sample_every_accepted_event() {
    let symbol = Symbol::new("average-overflow").unwrap();
    let raw = FeatureExtractorSpec::new([field(symbol, "p", EventField::Price)]).unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [
            T::standard_scale(id("p"), id("scaled"), 0., 0.5),
            T::lagged(id("p"), id("lag"), 1),
        ],
        [FittedStage::Scalar {
            transformations: vec![
                T::sma(id("scaled"), id("avg"), 2, WarmupPolicy::FullWindow),
                T::sma(id("lag"), id("lag_avg"), 2, WarmupPolicy::FullWindow),
            ],
        }],
        2,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<2>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(symbol, 1., 0), [f64::NAN, f64::NAN]),
        (Event::volume(symbol, 0., 1), [f64::NAN, f64::NAN]),
        (Event::volume(symbol, 0., 2), [f64::NAN, 1.]),
        (Event::price(symbol, 1e308, 3), [f64::NAN, 1.]),
        (Event::price(symbol, 3., 4), [4., 5e307]),
        (Event::volume(symbol, 0., 5), [4., 5e307]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &expected);
    }
}

#[test]
fn invoked_handlers_without_writes_do_not_sample() {
    let symbol = Symbol::new("average-cvd").unwrap();
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Cvd {
            symbol,
            source: fiml::FeatureSource::Event(fiml::EventKind::Trade),
            window: 1,
            warmup_policy: WarmupPolicy::FirstValue,
        },
        id("cvd"),
    )])
    .unwrap();
    let mut pipeline = PipelineSpec::new(
        raw,
        [T::sma(id("cvd"), id("avg"), 2, WarmupPolicy::FullWindow)],
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<1>::new(),
    )
    .unwrap();
    for (timestamp, side, volume, expected) in [
        (0, Some(fiml::TradeSide::AggressorBuy), 10., f64::NAN),
        (1, None, 30., f64::NAN),
        (2, Some(fiml::TradeSide::AggressorBuy), 20., 15.),
    ] {
        pipeline
            .handle_event(Event::trade(symbol, 1., volume, timestamp, side))
            .unwrap();
        equal(pipeline.values(), &[expected]);
    }
}

#[test]
fn imputed_values_and_missing_indicators_inherit_only_their_source_flags() {
    let a = Symbol::new("impute-average-a").unwrap();
    let b = Symbol::new("impute-average-b").unwrap();
    let raw = FeatureExtractorSpec::new([
        field(a, "a", EventField::Price),
        field(b, "b", EventField::Price),
    ])
    .unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [T::identity(id("a"), id("a")), T::identity(id("b"), id("b"))],
        [
            FittedStage::SimpleImpute {
                outputs: vec![id("a"), id("b"), id("missing")],
                retained_input_indices: vec![0, 1],
                replacement_values: vec![0., 0.],
                indicator_input_indices: vec![1],
            },
            FittedStage::Select {
                outputs: vec![id("b"), id("missing")],
                input_indices: vec![1, 2],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::sma(id("b"), id("b"), 2, WarmupPolicy::FullWindow),
                    T::ema(id("missing"), id("missing"), 2, WarmupPolicy::FullWindow),
                ],
            },
        ],
        2,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<2>::new(),
        ArrayFeatureVector::<2>::new(),
    )
    .unwrap();
    for (event, expected) in [
        (Event::price(a, 1., 0), [f64::NAN; 2]),
        (Event::price(b, 10., 1), [f64::NAN; 2]),
        (Event::price(a, 2., 2), [f64::NAN; 2]),
        (Event::price(b, 20., 3), [15., 0.]),
    ] {
        pipeline.handle_event(event).unwrap();
        equal(pipeline.values(), &expected);
    }
}

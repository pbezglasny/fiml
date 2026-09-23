use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractor,
    FeatureExtractorSpec, FeatureId, FeatureKey, FeatureSource, FeatureVector, FittedStage,
    PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
    order_book::{OrderBook, OrderBookLevel, OrderBookSnapshot, UpdatePolicy},
};
use rust_decimal::dec;

fn context(symbol: Symbol, name: &str, id: &str) -> FeatureDefinition {
    FeatureDefinition::new(
        FeatureKey::Context {
            symbol,
            name: name.into(),
        },
        FeatureId::new(id),
    )
}
fn id(name: &str) -> FeatureId {
    FeatureId::new(name)
}
fn same(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn context_is_atomic_partial_persistent_and_independent_of_books_and_time() {
    let btc = Symbol::new("ctx-btc").unwrap();
    let eth = Symbol::new("ctx-eth").unwrap();
    let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<5>::new())
        .add_feature(context(btc, "high", "btc_high"))
        .add_feature(FeatureDefinition::new(
            FeatureKey::OrderBookBestBidPrice { symbol: btc },
            id("bid"),
        ))
        .add_feature(context(eth, "high", "eth_high"))
        .add_feature(context(Symbol::GLOBAL, "volume", "volume"))
        .add_feature(FeatureDefinition::new(
            FeatureKey::DayOfWeek {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::AnyEvent,
            },
            id("day"),
        ))
        .add_order_book(btc, OrderBook::new(UpdatePolicy::Monotonic, 0))
        .build()
        .unwrap();
    assert!(
        extractor
            .feature_vector()
            .values()
            .iter()
            .all(|v| v.is_nan())
    );
    extractor
        .update_context(&[
            ("btc_high", Some(105.)),
            ("eth_high", Some(200.)),
            ("volume", Some(10.)),
        ])
        .unwrap();
    assert_eq!(extractor.last_timestamp(), None);
    extractor
        .handle_event(Event::order_book_snapshot(
            btc,
            0,
            OrderBookSnapshot::new(1, vec![OrderBookLevel::new(dec!(100), dec!(2))], vec![]),
        ))
        .unwrap();
    same(
        extractor.feature_vector().values(),
        &[105., 100., 200., 10., 4.],
    );
    let before = extractor.feature_vector().values().to_vec();
    for invalid in [
        vec![("missing", Some(1.))],
        vec![("bid", Some(1.))],
        vec![("btc_high", Some(f64::NAN))],
        vec![("eth_high", Some(f64::INFINITY))],
        vec![("volume", Some(f64::NEG_INFINITY))],
        vec![("btc_high", None)],
    ] {
        let mut updates = vec![("btc_high", Some(999.))];
        updates.extend(invalid);
        assert!(extractor.update_context(&updates).is_err());
        same(extractor.feature_vector().values(), &before);
    }
    extractor
        .update_context(&[("eth_high", None), ("volume", Some(20.))])
        .unwrap();
    extractor.update_context(&[]).unwrap();
    same(
        extractor.feature_vector().values(),
        &[105., 100., f64::NAN, 20., 4.],
    );
    assert_eq!(extractor.last_timestamp(), Some(0));
    assert_eq!(extractor.last_timestamp_for_symbol(btc), Some(0));
    assert_eq!(extractor.last_timestamp_for_symbol(eth), None);
    assert!(extractor.order_book_of_symbol(btc).is_some());
    extractor.handle_event(Event::time(86_400_000)).unwrap();
    same(
        extractor.feature_vector().values(),
        &[105., 100., f64::NAN, 20., 5.],
    );
}

#[test]
fn context_definitions_validate_names_keys_ids_and_reserved_ids() {
    for definitions in [
        vec![context(Symbol::GLOBAL, "", "x")],
        vec![
            context(Symbol::GLOBAL, "high", "x"),
            context(Symbol::GLOBAL, "high", "y"),
        ],
        vec![
            context(Symbol::GLOBAL, "high", "x"),
            context(Symbol::GLOBAL, "low", "x"),
        ],
        vec![context(Symbol::GLOBAL, "high", "__reserved_0")],
    ] {
        let mut builder = FeatureExtractor::builder(fiml::VecFeatureVector::new(definitions.len()));
        for definition in definitions {
            builder = builder.add_feature(definition);
        }
        assert!(builder.build().is_err());
    }
    let key = FeatureKey::Context {
        symbol: Symbol::GLOBAL,
        name: "previous_day_high".into(),
    };
    assert_eq!(
        FeatureId::from(&key).as_str(),
        "context:symbol=10:__global__:source=context:name=17:previous_day_high"
    );
}

#[test]
fn context_refresh_preserves_averages_and_event_lags_through_reused_stage_buffers() {
    let symbol = Symbol::new("ctx-price").unwrap();
    let raw = FeatureExtractorSpec::new([
        context(symbol, "high", "high"),
        FeatureDefinition::new(
            FeatureKey::Field {
                symbol,
                field: EventField::Price,
            },
            id("price"),
        ),
    ])
    .unwrap();
    let defs = vec![
        T::identity(id("high"), id("high")),
        T::lagged(id("high"), id("lag"), 1),
        T::sma(id("price"), id("sma"), 2, WarmupPolicy::FullWindow),
        T::ema(id("price"), id("ema"), 2, WarmupPolicy::FullWindow),
        T::sma(id("high"), id("context_sma"), 1, WarmupPolicy::FirstValue),
        T::ema(id("high"), id("context_ema"), 1, WarmupPolicy::FirstValue),
    ];
    for staged in [false, true] {
        let mut stages = vec![];
        if staged {
            // Three scalar stages force scratch reuse. Each lag needs retained output storage.
            stages.push(FittedStage::Scalar {
                transformations: vec![
                    T::identity(id("high"), id("high")),
                    T::lagged(id("high"), id("lag"), 1),
                    T::identity(id("sma"), id("sma")),
                    T::identity(id("ema"), id("ema")),
                    T::identity(id("context_sma"), id("context_sma")),
                    T::identity(id("context_ema"), id("context_ema")),
                ],
            });
            for _ in 0..2 {
                stages.push(FittedStage::Scalar {
                    transformations: ["high", "lag", "sma", "ema", "context_sma", "context_ema"]
                        .map(|name| T::identity(id(name), id(name)))
                        .to_vec(),
                });
            }
        }
        let mut pipeline = PipelineSpec::with_stages(raw.clone(), defs.clone(), stages, 6, None)
            .unwrap()
            .build(
                ArrayFeatureVector::<2>::new(),
                ArrayFeatureVector::<6>::new(),
            )
            .unwrap();
        pipeline.update_context(&[("high", Some(100.))]).unwrap();
        same(
            pipeline.values(),
            &[100., f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        );
        pipeline.handle_event(Event::price(symbol, 10., 1)).unwrap();
        pipeline.handle_event(Event::price(symbol, 20., 2)).unwrap();
        let before = pipeline.values().to_vec();
        assert_eq!(before[1], 100.);
        assert_eq!(before[2], 15.);
        for high in [200., 300.] {
            pipeline.update_context(&[("high", Some(high))]).unwrap();
            same(&pipeline.values()[1..], &before[1..]);
            assert_eq!(pipeline.values()[0], high);
            assert_eq!(pipeline.last_timestamp(), Some(2));
        }
        let unchanged = pipeline.values().to_vec();
        assert!(
            pipeline
                .update_context(&[("high", Some(9.)), ("price", Some(9.))])
                .is_err()
        );
        same(pipeline.values(), &unchanged);
        pipeline.handle_event(Event::price(symbol, 30., 3)).unwrap();
        assert_eq!(pipeline.values()[1], 100.);
        assert_eq!(pipeline.values()[2], 25.);
        pipeline.handle_event(Event::price(symbol, 40., 4)).unwrap();
        assert_eq!(pipeline.values()[1], 300.);
        assert!(pipeline.values()[4..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn context_refresh_runs_stateless_stages_and_clearing_without_events() {
    let raw = FeatureExtractorSpec::new([context(Symbol::GLOBAL, "high", "high")]).unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [T::standard_scale(id("high"), id("x"), 10., 2.)],
        [
            FittedStage::SimpleImpute {
                outputs: vec![id("x")],
                retained_input_indices: vec![0],
                replacement_values: vec![0.],
                indicator_input_indices: vec![],
            },
            FittedStage::StandardScale {
                outputs: vec![id("x")],
                mean: vec![1.],
                scale: vec![2.],
            },
            FittedStage::Select {
                outputs: vec![id("x")],
                input_indices: vec![0],
            },
            FittedStage::Pca {
                outputs: vec![id("pc")],
                mean: vec![1.],
                components: vec![vec![2.]],
                output_scale: vec![1.],
            },
        ],
        1,
        None,
    )
    .unwrap()
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<1>::new(),
    )
    .unwrap();
    pipeline.update_context(&[]).unwrap();
    assert!(pipeline.values()[0].is_nan());
    pipeline.update_context(&[("high", Some(20.))]).unwrap();
    assert_eq!(pipeline.values(), &[2.]);
    pipeline.update_context(&[("high", None)]).unwrap();
    assert_eq!(pipeline.values(), &[-3.]);
    assert_eq!(pipeline.last_timestamp(), None);
}

#[cfg(feature = "serde")]
#[test]
fn context_json_is_canonical_and_rebuilds_cold() {
    let definitions = [
        context(Symbol::new("ctx-json").unwrap(), "high", "high"),
        context(Symbol::GLOBAL, "low", "low"),
        FeatureDefinition::with_default_id(FeatureKey::Context {
            symbol: Symbol::GLOBAL,
            name: "volume".into(),
        }),
    ];
    let spec = FeatureExtractorSpec::new(definitions.clone()).unwrap();
    let reverse = FeatureExtractorSpec::new(definitions.into_iter().rev()).unwrap();
    assert_eq!(spec, reverse);
    let document = serde_json::to_value(&spec).unwrap();
    assert_eq!(document["version"], "2.0");
    assert_eq!(document["required_events"], serde_json::json!([]));
    let restored: FeatureExtractorSpec = serde_json::from_value(document).unwrap();
    assert_eq!(spec, restored);
    let mut runtime = spec.build(ArrayFeatureVector::<3>::new()).unwrap();
    runtime.update_context(&[("high", Some(10.))]).unwrap();
    assert!(
        restored
            .build(ArrayFeatureVector::<3>::new())
            .unwrap()
            .feature_vector()
            .values()
            .iter()
            .all(|v| v.is_nan())
    );
    let pipeline = PipelineSpec::new(restored, [T::identity(id("high"), id("out"))]).unwrap();
    let value = serde_json::to_value(&pipeline).unwrap();
    assert_eq!(value["version"], "3.0");
    assert_eq!(
        serde_json::from_value::<PipelineSpec>(value).unwrap(),
        pipeline
    );
}

#[cfg(feature = "serde")]
#[test]
fn shared_python_rust_context_replay_fixture() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/context_replay.json")).unwrap();
    let spec: PipelineSpec = serde_json::from_value(fixture["pipeline"].clone()).unwrap();
    let mut pipeline = spec
        .build(
            ArrayFeatureVector::<2>::new(),
            ArrayFeatureVector::<4>::new(),
        )
        .unwrap();
    for step in fixture["steps"].as_array().unwrap() {
        let updates: Vec<_> = step["context"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, value)| (id.as_str(), value.as_f64()))
            .collect();
        pipeline.update_context(&updates).unwrap();
        pipeline
            .handle_event(Event::price(
                Symbol::new("btc").unwrap(),
                step["price"].as_f64().unwrap(),
                step["timestamp"].as_i64().unwrap(),
            ))
            .unwrap();
        let expected: Vec<_> = step["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect();
        same(pipeline.values(), &expected);
    }
}

#[test]
fn downstream_averages_of_event_lags_do_not_sample_context_refreshes() {
    let raw = FeatureExtractorSpec::new([context(Symbol::GLOBAL, "high", "high")]).unwrap();
    let mut pipeline = PipelineSpec::with_stages(
        raw,
        [T::lagged(id("high"), id("lag"), 1)],
        [
            FittedStage::Scalar {
                transformations: vec![
                    T::sma(id("lag"), id("sma"), 2, WarmupPolicy::FullWindow),
                    T::ema(id("lag"), id("ema"), 2, WarmupPolicy::FullWindow),
                ],
            },
            FittedStage::Scalar {
                transformations: vec![
                    T::identity(id("sma"), id("sma")),
                    T::identity(id("ema"), id("ema")),
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
    pipeline.update_context(&[("high", Some(100.))]).unwrap();
    for timestamp in 0..3 {
        pipeline.handle_event(Event::time(timestamp)).unwrap();
    }
    same(pipeline.values(), &[100., 100.]);
    for high in [200., 300., 400.] {
        pipeline.update_context(&[("high", Some(high))]).unwrap();
        same(pipeline.values(), &[100., 100.]);
    }
    pipeline.handle_event(Event::time(3)).unwrap();
    same(pipeline.values(), &[100., 100.]);
    pipeline.handle_event(Event::time(4)).unwrap();
    assert_eq!(pipeline.values()[0], 250.);
    assert!((pipeline.values()[1] - 300.).abs() < 1e-12);
}

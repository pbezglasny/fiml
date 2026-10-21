use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use fiml::{
    Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey,
    FeatureSource, FeatureVector, PipelineSpec, Symbol, TransformerDefinition as T,
    VecFeatureVector, WarmupPolicy,
};
use std::{hint::black_box, ops::Range, time::Duration};

const ROUNDS_PER_BATCH: usize = 64;

fn price(symbol_index: usize, round: usize) -> f64 {
    100.0 + 10.0 * symbol_index as f64 + (round % 16) as f64
}

fn events_per_round(symbols: &[Symbol]) -> usize {
    3 * symbols.len() + 3
}

fn events(symbols: &[Symbol], unrelated: Symbol, rounds: Range<usize>) -> Vec<Event> {
    let stride = events_per_round(symbols);
    let mut events = Vec::with_capacity(rounds.len() * stride);
    for round in rounds {
        let start = (round * stride) as i64;
        for (index, &symbol) in symbols.iter().enumerate() {
            let timestamp = start + (3 * index) as i64;
            events.extend([
                Event::price(symbol, price(index, round), timestamp),
                Event::volume(symbol, 1.0 + (round % 4) as f64, timestamp + 1),
                Event::trade(symbol, price(index, round) + 0.5, 2.0, timestamp + 2, None),
            ]);
        }
        let timestamp = start + (3 * symbols.len()) as i64;
        events.extend([
            Event::time(timestamp),
            Event::price(unrelated, 10_000.0, timestamp + 1),
            Event::trade(unrelated, 20_000.0, 100.0, timestamp + 2, None),
        ]);
    }
    events
}

fn pipeline_spec(symbols: &[Symbol], windows: &[usize]) -> PipelineSpec {
    let aggregation = Duration::from_millis(events_per_round(symbols) as u64);
    let mut definitions = vec![FeatureDefinition::new(
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
            utc_offset_millis: 0,
        },
        FeatureId::new("elapsed"),
    )];
    for (index, &symbol) in symbols.iter().enumerate() {
        definitions.push(FeatureDefinition::new(
            FeatureKey::Field {
                symbol,
                field: EventField::Price,
            },
            FeatureId::new(format!("s{index}.price")),
        ));
        for &window in windows {
            definitions.extend([
                FeatureDefinition::new(
                    FeatureKey::SmaTimed {
                        symbol,
                        source: EventField::Price,
                        aggregation,
                        window: aggregation * window as u32,
                        warmup_policy: WarmupPolicy::FullWindow,
                    },
                    FeatureId::new(format!("s{index}.timed_sma{window}")),
                ),
                FeatureDefinition::new(
                    FeatureKey::TradeCountTimed {
                        symbol,
                        aggregation,
                        window: aggregation * window as u32,
                        warmup_policy: WarmupPolicy::FullWindow,
                    },
                    FeatureId::new(format!("s{index}.count{window}")),
                ),
            ]);
        }
    }
    let raw = FeatureExtractorSpec::new(definitions).unwrap();
    let mut transformations = Vec::new();
    for definition in raw.definitions() {
        let id = &definition.id;
        transformations.extend([
            T::identity(id.clone(), id.clone()),
            T::standard_scale(
                id.clone(),
                FeatureId::new(format!("{}.scaled", id.as_str())),
                100.0,
                10.0,
            ),
        ]);
    }
    for index in 0..symbols.len() {
        let input = FeatureId::new(format!("s{index}.price"));
        for &window in windows {
            transformations.extend([
                T::sma(
                    input.clone(),
                    FeatureId::new(format!("s{index}.sma{window}")),
                    window,
                    WarmupPolicy::FullWindow,
                ),
                T::ema(
                    input.clone(),
                    FeatureId::new(format!("s{index}.ema{window}")),
                    window,
                    WarmupPolicy::FullWindow,
                ),
            ]);
        }
    }
    // Several lags of one input exercise the compiler's shared history group.
    for &window in windows {
        transformations.push(T::lagged(
            FeatureId::new("elapsed"),
            FeatureId::new(format!("lag{window}")),
            window,
        ));
    }
    PipelineSpec::new(raw, transformations).unwrap()
}

fn assert_value(ids: &[FeatureId], values: &[f64], name: &str, expected: f64) {
    let index = ids.iter().position(|id| id.as_str() == name).unwrap();
    assert!(
        (values[index] - expected).abs() < 1e-9,
        "{name}: expected {expected}, got {}",
        values[index],
    );
}

fn measure_dispatch(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: BenchmarkId,
    symbols: &[Symbol],
    unrelated: Symbol,
    mut next_round: usize,
    mut dispatch: impl FnMut(Event),
) {
    group.throughput(Throughput::Elements(
        (ROUNDS_PER_BATCH * events_per_round(symbols)) as u64,
    ));
    group.bench_function(id, |b| {
        b.iter_batched_ref(
            || {
                let start = next_round;
                next_round += ROUNDS_PER_BATCH;
                events(symbols, unrelated, start..next_round)
            },
            |events| {
                for event in events.drain(..) {
                    dispatch(black_box(event));
                }
            },
            // Event creation and Vec destruction stay outside the timed batch.
            BatchSize::PerIteration,
        );
    });
}

fn compiled_dispatch(c: &mut Criterion) {
    // Reuse these names across cases; the process-wide symbol limit is 512.
    let symbols: Vec<_> = (0..16)
        .map(|index| Symbol::new(&format!("dispatch-{index}")).unwrap())
        .collect();
    let unrelated = Symbol::new("dispatch-unrelated").unwrap();
    let mut group = c.benchmark_group("compiled_dispatch");
    for (symbol_count, windows) in [
        (1, &[8][..]),
        (4, &[8, 32, 128][..]),
        (16, &[8, 32, 128, 512][..]),
    ] {
        let symbols = &symbols[..symbol_count];
        let spec = pipeline_spec(symbols, windows);
        let raw_spec = spec.raw_feature_extractor_spec();
        let raw_count = raw_spec.feature_vector_length();
        let model_count = spec.feature_vector_length();
        let mut extractor = raw_spec.build(VecFeatureVector::new(raw_count)).unwrap();
        let mut pipeline = spec
            .build(
                VecFeatureVector::new(raw_count),
                VecFeatureVector::new(model_count),
            )
            .unwrap();

        let max_window = *windows.last().unwrap();
        let warmup_rounds = 2 * max_window + 1;
        for event in events(symbols, unrelated, 0..warmup_rounds) {
            extractor.handle_event(event).unwrap();
        }
        for event in events(symbols, unrelated, 0..warmup_rounds) {
            pipeline.handle_event(event).unwrap();
        }

        let raw_ids = extractor.feature_ids();
        let raw_values = extractor.feature_vector().values();
        let output_ids = pipeline.output_ids();
        let values = pipeline.values();
        assert!(raw_values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| value.is_finite()));
        assert_eq!(raw_values, pipeline.raw_values());
        let elapsed = (warmup_rounds * events_per_round(symbols) - 1) as f64;
        assert_value(raw_ids, raw_values, "elapsed", elapsed);
        for (id, &value) in raw_ids.iter().zip(raw_values) {
            assert_value(output_ids, values, id.as_str(), value);
            assert_value(
                output_ids,
                values,
                &format!("{}.scaled", id.as_str()),
                (value - 100.0) / 10.0,
            );
        }
        for index in 0..symbol_count {
            assert_value(
                raw_ids,
                raw_values,
                &format!("s{index}.price"),
                price(index, warmup_rounds - 1),
            );
            for &window in windows {
                let mean = (warmup_rounds - window..warmup_rounds)
                    .map(|round| price(index, round))
                    .sum::<f64>()
                    / window as f64;
                assert_value(
                    raw_ids,
                    raw_values,
                    &format!("s{index}.timed_sma{window}"),
                    mean,
                );
                assert_value(
                    raw_ids,
                    raw_values,
                    &format!("s{index}.count{window}"),
                    window as f64,
                );
                assert_value(output_ids, values, &format!("s{index}.sma{window}"), mean);
            }
        }
        for &window in windows {
            assert_value(
                output_ids,
                values,
                &format!("lag{window}"),
                elapsed - window as f64,
            );
        }

        let case = format!(
            "symbols={symbol_count},windows={},max_window={max_window},raw={raw_count}",
            windows.len(),
        );
        measure_dispatch(
            &mut group,
            BenchmarkId::new("extractor", &case),
            symbols,
            unrelated,
            warmup_rounds,
            |event| {
                black_box(extractor.handle_event(event).unwrap());
                black_box(extractor.feature_vector().values());
            },
        );
        measure_dispatch(
            &mut group,
            BenchmarkId::new("pipeline", format!("{case},model={model_count}")),
            symbols,
            unrelated,
            warmup_rounds,
            |event| {
                black_box(pipeline.handle_event(event).unwrap());
                black_box(pipeline.values());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, compiled_dispatch);
criterion_main!(benches);

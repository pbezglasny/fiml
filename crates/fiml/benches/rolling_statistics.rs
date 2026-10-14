use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};
use std::{hint::black_box, time::Duration};

fn rolling_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_statistics");
    for mode in ["sample", "timed", "trajectory"] {
        for window in [100, 10_000, 100_000] {
            let id = FeatureId::new;
            let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
                FeatureKey::Field {
                    symbol: Symbol::GLOBAL,
                    field: EventField::Price,
                },
                id("p"),
            )])
            .unwrap();
            let warmup = WarmupPolicy::FirstValue;
            let definitions = if mode == "trajectory" {
                vec![
                    T::rolling_slope(id("p"), id("slope"), window, warmup),
                    T::rolling_path_length(id("p"), id("path_length"), window, warmup),
                    T::rolling_positive_movement(id("p"), id("positive_movement"), window, warmup),
                    T::rolling_negative_movement(id("p"), id("negative_movement"), window, warmup),
                    T::rolling_direction_change_count(
                        id("p"),
                        id("direction_change_count"),
                        window,
                        warmup,
                    ),
                    T::rolling_max_drawdown(id("p"), id("max_drawdown"), window, warmup),
                ]
            } else if mode == "timed" {
                let window = Duration::from_millis(window as u64);
                vec![
                    T::rolling_std_timed(id("p"), id("std"), window, warmup),
                    T::rolling_min_timed(id("p"), id("min"), window, warmup),
                    T::rolling_max_timed(id("p"), id("max"), window, warmup),
                    T::rolling_range_timed(id("p"), id("range"), window, warmup),
                ]
            } else {
                vec![
                    T::rolling_std(id("p"), id("std"), window, warmup),
                    T::rolling_min(id("p"), id("min"), window, warmup),
                    T::rolling_max(id("p"), id("max"), window, warmup),
                    T::rolling_range(id("p"), id("range"), window, warmup),
                ]
            };
            let length = definitions.len();
            let mut pipeline = PipelineSpec::with_capacity(raw, definitions, 6)
                .unwrap()
                .build(
                    ArrayFeatureVector::<1>::new(),
                    ArrayFeatureVector::<6>::new_of_length(length),
                )
                .unwrap();
            let mut time = 0_i64;
            for _ in 0..window {
                pipeline
                    .handle_event(Event::price(Symbol::GLOBAL, (time % 17) as f64, time))
                    .unwrap();
                time += 1;
            }
            group.bench_function(BenchmarkId::new(mode, window), |b| {
                b.iter(|| {
                    pipeline
                        .handle_event(Event::price(Symbol::GLOBAL, (time % 17) as f64, time))
                        .unwrap();
                    time += 1;
                    black_box(pipeline.values());
                })
            });
        }
    }
    group.finish();
}
criterion_group!(benches, rolling_statistics);
criterion_main!(benches);

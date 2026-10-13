use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};
use std::{hint::black_box, time::Duration};

fn rolling_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_statistics");
    for timed in [false, true] {
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
            let definitions = if timed {
                let window = Duration::from_millis(window as u64);
                [
                    T::rolling_std_timed(id("p"), id("std"), window, warmup),
                    T::rolling_min_timed(id("p"), id("min"), window, warmup),
                    T::rolling_max_timed(id("p"), id("max"), window, warmup),
                    T::rolling_range_timed(id("p"), id("range"), window, warmup),
                ]
            } else {
                [
                    T::rolling_std(id("p"), id("std"), window, warmup),
                    T::rolling_min(id("p"), id("min"), window, warmup),
                    T::rolling_max(id("p"), id("max"), window, warmup),
                    T::rolling_range(id("p"), id("range"), window, warmup),
                ]
            };
            let mut pipeline = PipelineSpec::new(raw, definitions)
                .unwrap()
                .build(
                    ArrayFeatureVector::<1>::new(),
                    ArrayFeatureVector::<4>::new(),
                )
                .unwrap();
            let mut time = 0_i64;
            for _ in 0..window {
                pipeline
                    .handle_event(Event::price(Symbol::GLOBAL, (time % 17) as f64, time))
                    .unwrap();
                time += 1;
            }
            group.bench_function(
                BenchmarkId::new(if timed { "timed" } else { "sample" }, window),
                |b| {
                    b.iter(|| {
                        pipeline
                            .handle_event(Event::price(Symbol::GLOBAL, (time % 17) as f64, time))
                            .unwrap();
                        time += 1;
                        black_box(pipeline.values());
                    })
                },
            );
        }
    }
    group.finish();
}
criterion_group!(benches, rolling_statistics);
criterion_main!(benches);

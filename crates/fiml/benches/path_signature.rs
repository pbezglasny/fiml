use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fiml::{
    Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey, FittedStage,
    PipelineSpec, Symbol, TransformerDefinition as T, WarmupPolicy,
};
use std::hint::black_box;

fn path_signature(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_path_signature");
    for window in [2, 100, 10_000, 100_000] {
        let id = FeatureId::new;
        let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
            FeatureKey::Field {
                symbol: Symbol::GLOBAL,
                field: EventField::Price,
            },
            id("p"),
        )])
        .unwrap();
        let mut pipeline = {
            let mut spec = PipelineSpec::new(raw);
            let mut node = spec
                .add(
                    spec.root(),
                    fiml::FittedStage::Scalar {
                        transformations: [T::identity(id("p"), id("p"))].into_iter().collect(),
                    },
                )
                .unwrap();
            node = spec
                .add(
                    node,
                    FittedStage::rolling_path_signature(
                        id("p"),
                        "sig",
                        window,
                        WarmupPolicy::FullWindow,
                    ),
                )
                .unwrap();
            spec.output("model", node).unwrap();
            spec
        }
        .build()
        .unwrap();
        let mut time = 0_i64;
        for _ in 0..window {
            pipeline
                .handle_event(Event::price(Symbol::GLOBAL, (time % 17) as f64, time))
                .unwrap();
            time += 1;
        }
        group.bench_function(BenchmarkId::new("level_2", window), |b| {
            b.iter(|| {
                pipeline
                    .handle_event(Event::price(Symbol::GLOBAL, (time % 17) as f64, time))
                    .unwrap();
                time += 1;
                black_box(pipeline.output("model").unwrap().values());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, path_signature);
criterion_main!(benches);

//! Compares one shared preprocessing graph with two independently built pipelines.
use criterion::{Criterion, criterion_group, criterion_main};
use fiml::{
    Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey, FittedStage,
    PipelineSpec, Symbol, TransformerDefinition, WarmupPolicy,
};
use std::hint::black_box;

fn spec(branches: usize) -> PipelineSpec {
    let id = FeatureId::new;
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Field {
            symbol: Symbol::GLOBAL,
            field: EventField::Price,
        },
        id("price"),
    )])
    .unwrap();
    let mut spec = PipelineSpec::new(raw);
    let shared = spec
        .add(
            spec.root(),
            FittedStage::Scalar {
                transformations: (1..=16)
                    .map(|window| {
                        TransformerDefinition::sma(
                            id("price"),
                            FeatureId::new(format!("mean{window}")),
                            window,
                            WarmupPolicy::FirstValue,
                        )
                    })
                    .collect(),
            },
        )
        .unwrap();
    let projected = spec
        .add(
            shared,
            FittedStage::Pca {
                outputs: (0..8).map(|i| FeatureId::new(format!("pc{i}"))).collect(),
                mean: vec![10.; 16],
                components: vec![vec![1. / 16.; 16]; 8],
                output_scale: vec![1.; 8],
            },
        )
        .unwrap();
    for branch in 0..branches {
        let node = spec
            .add(
                projected,
                FittedStage::Scalar {
                    transformations: vec![TransformerDefinition::lagged(id("pc0"), id("lag"), 3)],
                },
            )
            .unwrap();
        spec.output(format!("model{branch}"), node).unwrap();
    }
    spec
}

fn compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_outputs");
    let mut shared = spec(2).build().unwrap();
    let ids = [
        shared.output("model0").unwrap().id(),
        shared.output("model1").unwrap().id(),
    ];
    let mut timestamp = 0;
    group.bench_function("shared_graph", |b| {
        b.iter(|| {
            timestamp += 1;
            shared
                .handle_event(Event::price(
                    Symbol::GLOBAL,
                    black_box((timestamp % 100) as f64),
                    timestamp,
                ))
                .unwrap();
            for id in ids {
                black_box(shared.output_by_id(id).unwrap().values());
            }
        })
    });
    let mut independent = [spec(1).build().unwrap(), spec(1).build().unwrap()];
    let ids = independent
        .each_ref()
        .map(|p| p.output("model0").unwrap().id());
    let mut timestamp = 0;
    group.bench_function("independent_pipelines", |b| {
        b.iter(|| {
            timestamp += 1;
            for (pipeline, id) in independent.iter_mut().zip(ids) {
                pipeline
                    .handle_event(Event::price(
                        Symbol::GLOBAL,
                        black_box((timestamp % 100) as f64),
                        timestamp,
                    ))
                    .unwrap();
                black_box(pipeline.output_by_id(id).unwrap().values());
            }
        })
    });
    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);

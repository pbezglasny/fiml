use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fiml::{
    Event, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey, FeatureSource,
    FittedStage, Pipeline, PipelineSpec, Symbol, TransformerDefinition, VecFeatureVector,
};

const ROWS: usize = 10_000;

fn pipeline(
    dimensions: usize,
    quantile_count: usize,
) -> Pipeline<VecFeatureVector, VecFeatureVector> {
    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        },
        FeatureId::new("day"),
    )])
    .unwrap();
    let outputs = (0..dimensions)
        .map(|index| FeatureId::new(format!("f{index}")))
        .collect::<Vec<_>>();
    let transformations = outputs
        .iter()
        .cloned()
        .map(|output| TransformerDefinition::identity(FeatureId::new("day"), output))
        .collect::<Vec<_>>();
    let references = (0..quantile_count)
        .map(|index| index as f64 / (quantile_count - 1) as f64)
        .collect::<Vec<_>>();
    let quantiles = references
        .iter()
        .map(|reference| vec![reference * 6.0; dimensions])
        .collect();
    PipelineSpec::with_stages(
        raw,
        transformations,
        [FittedStage::QuantileTransform {
            outputs,
            output_distribution: "normal".into(),
            quantiles,
            references,
            all_nan_input_indices: vec![],
            bounds_threshold: 1e-7,
            normal_clip: Some((-5.199_337_582_605_575, 5.199_337_582_703_42)),
        }],
        dimensions,
        None,
    )
    .unwrap()
    .build(VecFeatureVector::new(1), VecFeatureVector::new(dimensions))
    .unwrap()
}

fn bench_quantile_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile_transform");
    group.throughput(Throughput::Elements(ROWS as u64));
    for (dimensions, quantiles) in [(8, 64), (32, 256)] {
        group.bench_function(
            BenchmarkId::new("normal", format!("d={dimensions},q={quantiles}")),
            |b| {
                b.iter_batched(
                    || pipeline(dimensions, quantiles),
                    |mut pipeline| {
                        for day in 0..ROWS {
                            black_box(
                                pipeline
                                    .handle_event(Event::time(day as i64 * 86_400_000))
                                    .unwrap(),
                            );
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_quantile_transform);
criterion_main!(benches);

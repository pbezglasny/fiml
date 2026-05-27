use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fiml::{HeapRingBuffer, SimpleMovingAverage, StackRingBuffer};
use std::hint::black_box;

const N_POINTS: usize = 10_000;

fn gen_inputs(n: usize) -> Vec<f64> {
    // Simple LCG so runs are reproducible without pulling in `rand`.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f64) / (u32::MAX as f64) * 100.0
        })
        .collect()
}

fn bench_sma_update(c: &mut Criterion) {
    let input = gen_inputs(N_POINTS);

    let mut group = c.benchmark_group("sma_update");
    group.throughput(Throughput::Elements(N_POINTS as u64));

    macro_rules! stack_case {
        ($period:literal, $windows:literal) => {{
            let id = BenchmarkId::new("stack", format!("period={},windows={}", $period, $windows));
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut sma: SimpleMovingAverage<StackRingBuffer<$period, f64>, f64, $windows> =
                        SimpleMovingAverage::new_stack();
                    for w in 1..=$windows {
                        sma.add_window(w * ($period / $windows).max(1)).unwrap();
                    }
                    for &v in input.iter() {
                        sma.update(black_box(v));
                    }
                    black_box(sma.values())
                });
            });
        }};
    }

    stack_case!(20, 1);
    stack_case!(20, 3);
    stack_case!(200, 1);
    stack_case!(200, 3);

    macro_rules! heap_case {
        ($period:literal, $windows:literal) => {{
            let id = BenchmarkId::new("heap", format!("period={},windows={}", $period, $windows));
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, $windows> =
                        SimpleMovingAverage::new_heap($period);
                    for w in 1..=$windows {
                        sma.add_window(w * ($period / $windows).max(1)).unwrap();
                    }
                    for &v in input.iter() {
                        sma.update(black_box(v));
                    }
                    black_box(sma.values())
                });
            });
        }};
    }

    heap_case!(20, 1);
    heap_case!(20, 3);
    heap_case!(200, 1);
    heap_case!(200, 3);

    group.finish();
}

fn bench_sma_steady_state(c: &mut Criterion) {
    // Measure only steady-state updates (buffer already full → peek_back_at path).
    let input = gen_inputs(N_POINTS);

    let mut group = c.benchmark_group("sma_steady_state");
    group.throughput(Throughput::Elements(N_POINTS as u64));

    group.bench_function("stack_period200_windows3", |b| {
        b.iter_batched(
            || {
                let mut sma: SimpleMovingAverage<StackRingBuffer<200, f64>, f64, 3> =
                    SimpleMovingAverage::new_stack();
                sma.add_window(50).unwrap();
                sma.add_window(100).unwrap();
                sma.add_window(200).unwrap();
                // Warm to capacity.
                for &v in input.iter().take(200) {
                    sma.update(v);
                }
                sma
            },
            |mut sma| {
                for &v in input.iter() {
                    sma.update(black_box(v));
                }
                black_box(sma.values())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("heap_period200_windows3", |b| {
        b.iter_batched(
            || {
                let mut sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, 3> =
                    SimpleMovingAverage::new_heap(200);
                sma.add_window(50).unwrap();
                sma.add_window(100).unwrap();
                sma.add_window(200).unwrap();
                for &v in input.iter().take(200) {
                    sma.update(v);
                }
                sma
            },
            |mut sma| {
                for &v in input.iter() {
                    sma.update(black_box(v));
                }
                black_box(sma.values())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_sma_update, bench_sma_steady_state);
criterion_main!(benches);

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fiml::{HeapRingBuffer, RingBuffer, new_heap_ring_buffer, new_stack_ring_buffer};
use std::hint::black_box;

const OPS: usize = 10_000;

fn bench_push_back_steady(c: &mut Criterion) {
    // Buffer is full; every push overwrites the front (the realistic hot path).
    let mut group = c.benchmark_group("push_back_steady");
    group.throughput(Throughput::Elements(OPS as u64));

    macro_rules! stack_case {
        ($cap:literal) => {{
            group.bench_function(BenchmarkId::new("stack", $cap), |b| {
                b.iter_batched(
                    || {
                        let mut buf = new_stack_ring_buffer::<$cap, f64>();
                        for i in 0..$cap {
                            buf.push_back(i as f64);
                        }
                        buf
                    },
                    |mut buf| {
                        for i in 0..OPS {
                            black_box(buf.push_back(black_box(i as f64)));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        }};
    }

    stack_case!(8);
    stack_case!(64);
    stack_case!(1024);

    for &cap in &[8usize, 64, 1024] {
        group.bench_function(BenchmarkId::new("heap", cap), |b| {
            b.iter_batched(
                || {
                    let mut buf: HeapRingBuffer<f64> = new_heap_ring_buffer(cap);
                    for i in 0..cap {
                        buf.push_back(i as f64);
                    }
                    buf
                },
                |mut buf| {
                    for i in 0..OPS {
                        black_box(buf.push_back(black_box(i as f64)));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_push_back_fill(c: &mut Criterion) {
    // Cold path: empty → full, no overwrites.
    let mut group = c.benchmark_group("push_back_fill");

    macro_rules! stack_case {
        ($cap:literal) => {{
            group.throughput(Throughput::Elements($cap as u64));
            group.bench_function(BenchmarkId::new("stack", $cap), |b| {
                b.iter(|| {
                    let mut buf = new_stack_ring_buffer::<$cap, f64>();
                    for i in 0..$cap {
                        black_box(buf.push_back(black_box(i as f64)));
                    }
                    buf
                });
            });
        }};
    }

    stack_case!(8);
    stack_case!(64);
    stack_case!(1024);

    for &cap in &[8usize, 64, 1024] {
        group.throughput(Throughput::Elements(cap as u64));
        group.bench_function(BenchmarkId::new("heap", cap), |b| {
            b.iter(|| {
                let mut buf: HeapRingBuffer<f64> = new_heap_ring_buffer(cap);
                for i in 0..cap {
                    black_box(buf.push_back(black_box(i as f64)));
                }
                buf
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_push_back_steady, bench_push_back_fill);
criterion_main!(benches);

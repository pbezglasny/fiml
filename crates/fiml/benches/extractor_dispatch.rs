use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use fiml::{Event, FeatureExtractor, FeatureSet, IndicatorFeatures, ValueSource, symbols};

fn benchmark_dispatch(c: &mut Criterion) {
    let aapl = symbols::intern("AAPL");
    let msft = symbols::intern("MSFT");

    c.bench_function("extractor_dispatch/one_output", |benchmark| {
        let feature_set = FeatureSet::builder().sma("AAPL", [20]).build();
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set).unwrap();
        let mut timestamp = 0;
        benchmark.iter(|| {
            timestamp += 1;
            extractor
                .dispatch(black_box(&Event::price(aapl, 100.0, timestamp)))
                .unwrap();
        });
    });

    c.bench_function("extractor_dispatch/grouped_outputs", |benchmark| {
        let feature_set = FeatureSet::builder()
            .sma("AAPL", [5, 10, 20, 50, 100])
            .build();
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set).unwrap();
        let mut timestamp = 0;
        benchmark.iter(|| {
            timestamp += 1;
            extractor
                .dispatch(black_box(&Event::price(aapl, 100.0, timestamp)))
                .unwrap();
        });
    });

    c.bench_function("extractor_dispatch/multiple_symbols_routes", |benchmark| {
        let feature_set = FeatureSet::builder()
            .sma("AAPL", [20])
            .ema_from("MSFT", ValueSource::TradeVolume, [20])
            .build();
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set).unwrap();
        let mut timestamp = 0;
        benchmark.iter(|| {
            timestamp += 1;
            extractor
                .dispatch(black_box(&Event::price(aapl, 100.0, timestamp)))
                .unwrap();
            extractor
                .dispatch(black_box(&Event::trade(msft, 200.0, 10.0, timestamp, None)))
                .unwrap();
        });
    });

    // Time-decaying features are advanced on every dispatch, whatever the
    // event's kind or symbol, so this measures the refresh pass rather than the
    // indicator update it precedes.
    c.bench_function("extractor_dispatch/time_decaying_refresh", |benchmark| {
        let feature_set = FeatureSet::builder()
            .trade_count_timed("AAPL", Duration::from_millis(100), Duration::from_secs(60))
            .obv_timed(
                "AAPL",
                Duration::from_millis(100),
                [Duration::from_secs(60)],
            )
            .sma_timed_from(
                "AAPL",
                ValueSource::TradePrice,
                Duration::from_millis(100),
                [Duration::from_secs(60)],
            )
            .build();
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set).unwrap();
        let mut timestamp = 0;
        benchmark.iter(|| {
            timestamp += 1;
            extractor
                .dispatch(black_box(&Event::trade(aapl, 100.0, 1.0, timestamp, None)))
                .unwrap();
        });
    });

    // The same three indicators advanced by events that never feed them: every
    // dispatch pays the refresh pass and nothing else.
    c.bench_function(
        "extractor_dispatch/time_decaying_other_symbol",
        |benchmark| {
            let feature_set = FeatureSet::builder()
                .trade_count_timed("AAPL", Duration::from_millis(100), Duration::from_secs(60))
                .obv_timed(
                    "AAPL",
                    Duration::from_millis(100),
                    [Duration::from_secs(60)],
                )
                .sma_timed_from(
                    "AAPL",
                    ValueSource::TradePrice,
                    Duration::from_millis(100),
                    [Duration::from_secs(60)],
                )
                .build();
            let mut extractor = FeatureExtractor::from_feature_set(&feature_set).unwrap();
            let mut timestamp = 0;
            benchmark.iter(|| {
                timestamp += 1;
                extractor
                    .dispatch(black_box(&Event::trade(msft, 200.0, 1.0, timestamp, None)))
                    .unwrap();
            });
        },
    );

    c.bench_function("extractor_dispatch/clock", |benchmark| {
        let feature_set = FeatureSet::builder()
            .day_of_week()
            .time_since_first_event_of_day(0)
            .build();
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set).unwrap();
        let mut timestamp = 0;
        benchmark.iter(|| {
            timestamp += 1;
            extractor
                .dispatch(black_box(&Event::price(aapl, 100.0, timestamp)))
                .unwrap();
        });
    });
}

criterion_group!(benches, benchmark_dispatch);
criterion_main!(benches);

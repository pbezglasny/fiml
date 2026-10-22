mod common;

use std::time::Duration;

use fiml::indicators::{
    CountBucket, ObvBucket, OnBalanceVolumeTimed, RollingTradeVolumeTimed,
    SimpleMovingAverageTimed, TradeCountTimed, TradeVolumeBucket, VolumeWeightedAveragePriceTimed,
    VwapBucket,
};
use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureKey,
    FeatureVector, HeapRingBuffer, StackRingBuffer, Symbol, TransformerDefinition, WarmupPolicy,
};

fn assert_values(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            actual == expected || (actual.is_nan() && expected.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn timestamp_boundaries_match_in_standalone_extractor_and_pipeline() -> fiml::Result<()> {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/timed_timestamp_boundaries.json"
    ))
    .unwrap();
    let symbol = Symbol::new("BTC")?;
    for fixture in fixtures.as_array().unwrap() {
        let aggregation = Duration::from_millis(fixture["aggregation_ms"].as_u64().unwrap());
        let window = Duration::from_millis(fixture["window_ms"].as_u64().unwrap());
        let periods = (window.as_millis() / aggregation.as_millis()) as usize;
        for warmup_policy in [WarmupPolicy::FirstValue, WarmupPolicy::FullWindow] {
            let raw = FeatureExtractorSpec::new(
                [
                    FeatureKey::SmaTimed {
                        symbol,
                        source: EventField::TradePrice,
                        aggregation,
                        window,
                        warmup_policy,
                    },
                    FeatureKey::ObvTimed {
                        symbol,
                        aggregation,
                        window,
                        warmup_policy,
                    },
                    FeatureKey::TradeCountTimed {
                        symbol,
                        aggregation,
                        window,
                        warmup_policy,
                    },
                    FeatureKey::TradeVolumeTimed {
                        symbol,
                        aggregation,
                        window,
                        warmup_policy,
                    },
                    FeatureKey::VwapTimed {
                        symbol,
                        aggregation,
                        window,
                        warmup_policy,
                    },
                ]
                .map(FeatureDefinition::with_default_id),
            )?;
            let mut extractor = raw.build(ArrayFeatureVector::<5>::new())?;
            let transformations = raw
                .definitions()
                .iter()
                .map(|definition| {
                    TransformerDefinition::identity(definition.id.clone(), definition.id.clone())
                })
                .collect::<Vec<_>>();
            let mut pipeline = common::linear_spec(raw, transformations, [], None)?.build()?;
            let mut sma = SimpleMovingAverageTimed::<StackRingBuffer<3, (i64, f64)>, 1>::new_stack(
                aggregation,
                warmup_policy,
            )?;
            let mut obv = OnBalanceVolumeTimed::<HeapRingBuffer<ObvBucket>, 1>::new_heap(
                aggregation,
                3,
                warmup_policy,
            )?;
            let mut count = TradeCountTimed::<HeapRingBuffer<CountBucket>>::new_heap(
                aggregation,
                window,
                warmup_policy,
            )?;
            let mut volume =
                RollingTradeVolumeTimed::<StackRingBuffer<3, TradeVolumeBucket>, 1>::new_stack(
                    aggregation,
                    warmup_policy,
                )?;
            let mut vwap =
                VolumeWeightedAveragePriceTimed::<HeapRingBuffer<VwapBucket>, 1>::new_heap(
                    aggregation,
                    3,
                    warmup_policy,
                )?;
            sma.add_window_with_periods(periods)?;
            obv.add_window_with_periods(periods)?;
            volume.add_window_with_periods(periods)?;
            vwap.add_window_with_periods(periods)?;

            for step in fixture["steps"].as_array().unwrap() {
                let timestamp = step["timestamp"].as_i64().unwrap();
                if let Some(price) = step["price"].as_f64() {
                    let size = step["volume"].as_f64().unwrap();
                    sma.update(price, timestamp);
                    obv.update(price, size, timestamp);
                    count.update(timestamp);
                    volume.update(size, timestamp);
                    vwap.update(price, size, timestamp);
                } else {
                    sma.observe(timestamp);
                    obv.observe(timestamp);
                    count.observe(timestamp);
                    volume.observe(timestamp);
                    vwap.observe(timestamp);
                }
                let event = || {
                    if let Some(price) = step["price"].as_f64() {
                        Event::trade(
                            symbol,
                            price,
                            step["volume"].as_f64().unwrap(),
                            timestamp,
                            None,
                        )
                    } else {
                        Event::volume(symbol, 1.0, timestamp)
                    }
                };
                extractor.handle_event(event())?;
                pipeline.handle_event(event())?;
                let warming = warmup_policy == WarmupPolicy::FullWindow
                    && i128::from(timestamp) - i128::from(i64::MIN) < window.as_millis() as i128;
                let expected: Vec<_> = step["expected"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| {
                        if warming {
                            f64::NAN
                        } else {
                            value.as_f64().unwrap_or(f64::NAN)
                        }
                    })
                    .collect();
                assert_values(
                    &[
                        obv.window_value(0).unwrap_or(f64::NAN),
                        sma.value_at(0).unwrap_or(f64::NAN),
                        count.window_value().unwrap_or(f64::NAN),
                        volume.window_value(0).unwrap_or(f64::NAN),
                        vwap.window_value(0).unwrap_or(f64::NAN),
                    ],
                    &expected,
                );
                assert_values(extractor.feature_vector().values(), &expected);
                assert_values(pipeline.raw_values(), &expected);
                assert_values(pipeline.output("model").unwrap().values(), &expected);
            }
        }
    }
    Ok(())
}

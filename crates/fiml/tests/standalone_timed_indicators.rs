use std::time::Duration;

use fiml::indicators::{
    CountBucket, ObvBucket, OnBalanceVolumeTimed, RollingTradeVolumeTimed, RollingVolatilityTimed,
    SimpleMovingAverageTimed, TradeCountTimed, TradeVolumeBucket, VolatilityBucket,
    VolumeWeightedAveragePriceTimed, VwapBucket,
};
use fiml::{HeapRingBuffer, WarmupPolicy};

#[test]
fn standalone_timed_indicators_replay_and_expire_without_the_system_clock() -> fiml::Result<()> {
    let aggregation = Duration::from_secs(1);
    let warmup = WarmupPolicy::FullWindow;
    let mut count = TradeCountTimed::<HeapRingBuffer<CountBucket>>::new_heap(
        aggregation,
        Duration::from_secs(2),
        warmup,
    )?;
    let mut volume = RollingTradeVolumeTimed::<HeapRingBuffer<TradeVolumeBucket>, 1>::new_heap(
        aggregation,
        3,
        warmup,
    )?;
    let mut vwap = VolumeWeightedAveragePriceTimed::<HeapRingBuffer<VwapBucket>, 1>::new_heap(
        aggregation,
        3,
        warmup,
    )?;
    let mut obv =
        OnBalanceVolumeTimed::<HeapRingBuffer<ObvBucket>, 1>::new_heap(aggregation, 3, warmup)?;
    let mut sma = SimpleMovingAverageTimed::<HeapRingBuffer<(i64, f64)>, 1>::new_heap(
        aggregation,
        3,
        warmup,
    )?;
    let mut volatility = RollingVolatilityTimed::<HeapRingBuffer<VolatilityBucket>, 1>::new_heap(
        aggregation,
        3,
        warmup,
    )?;
    volume.add_window_with_periods(2)?;
    vwap.add_window_with_periods(2)?;
    obv.add_window_with_periods(2)?;
    sma.add_window_with_periods(2)?;
    volatility.add_window_with_periods(2)?;

    for (timestamp, price, size) in [(0, 10.0, 1.0), (1_000, 20.0, 2.0)] {
        count.update(timestamp);
        volume.update(size, timestamp);
        vwap.update(price, size, timestamp);
        obv.update(price, size, timestamp);
        sma.update(price, timestamp);
        volatility.update(price, timestamp);
    }
    assert_eq!(count.window_value(), None);
    assert_eq!(volume.window_value(0), None);
    assert_eq!(vwap.window_value(0), None);
    assert_eq!(obv.window_value(0), None);
    assert_eq!(sma.value_at(0), None);
    assert_eq!(volatility.value_at(0), None);

    // Time alone completes warm-up and expires the first bucket.
    assert!(count.observe(2_000));
    assert!(volume.observe(2_000));
    assert!(vwap.observe(2_000));
    assert!(obv.observe(2_000));
    assert!(sma.observe(2_000));
    assert!(volatility.observe(2_000));
    assert_eq!(count.window_value(), Some(1.0));
    assert_eq!(volume.window_value(0), Some(2.0));
    assert_eq!(vwap.window_value(0), Some(20.0));
    assert_eq!(obv.window_value(0), Some(2.0));
    assert_eq!(sma.value_at(0), Some(20.0));
    assert_eq!(volatility.value_at(0), Some(0.0));

    // An update at the already observed timestamp still records new data.
    count.update(2_000);
    volume.update(3.0, 2_000);
    vwap.update(10.0, 3.0, 2_000);
    obv.update(10.0, 3.0, 2_000);
    sma.update(10.0, 2_000);
    volatility.update(10.0, 2_000);
    assert_eq!(count.window_value(), Some(2.0));
    assert_eq!(volume.window_value(0), Some(5.0));
    assert_eq!(vwap.window_value(0), Some(14.0));
    assert_eq!(obv.window_value(0), Some(-1.0));
    assert_eq!(sma.value_at(0), Some(15.0));
    assert_eq!(volatility.value_at(0), Some(0.75));

    // Expiry is idempotent and does not synthesize a zero-price trade.
    for expected in [true, false] {
        assert_eq!(count.observe(4_000), expected);
        assert_eq!(volume.observe(4_000), expected);
        assert_eq!(vwap.observe(4_000), expected);
        assert_eq!(obv.observe(4_000), expected);
        assert_eq!(sma.observe(4_000), expected);
        assert_eq!(volatility.observe(4_000), expected);
        assert_eq!(count.window_value(), Some(0.0));
        assert_eq!(volume.window_value(0), Some(0.0));
        assert_eq!(vwap.window_value(0), None);
        assert_eq!(obv.window_value(0), Some(0.0));
        assert_eq!(sma.value_at(0), None);
        assert_eq!(volatility.value_at(0), None);
    }
    Ok(())
}

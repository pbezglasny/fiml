use std::time::Duration;

use fiml::indicators::{
    CumulativeVolumeDelta, ExponentialMovingAverage, ObvBucket, OnBalanceVolumeTimed,
    RollingTradeVolumeTimed, RollingVolatility, RollingVolatilityTimed, SimpleMovingAverage,
    SimpleMovingAverageTimed, TradeVolumeBucket, VolatilityBucket, VolumeWeightedAveragePriceTimed,
    VwapBucket,
};
use fiml::{HeapRingBuffer, StackRingBuffer, TradeSide, WarmupPolicy};

// Run the same public window-boundary checks with each history storage choice.
macro_rules! window_cases {
    (
        $name:ident, |$windows:ident, $warmup:ident, $indicator:ident|
        [$($constructor:expr),+ $(,)?],
        $add:ident, $read:ident, $update:block, $expected:expr
    ) => {
        #[test]
        fn $name() {
            fn check<const $windows: usize>() {
                for $warmup in [WarmupPolicy::FirstValue, WarmupPolicy::FullWindow] {
                    for active in 0..=$windows {
                        $({
                            let mut $indicator = $constructor;
                            assert!(!$indicator.is_ready());
                            for index in [0, $windows, usize::MAX] {
                                assert!(!$indicator.is_ready_at(index));
                                assert_eq!($indicator.$read(index), None);
                            }
                            for _ in 0..active {
                                $indicator.$add(1).unwrap();
                            }
                            assert!($indicator.$add(0).is_err());
                            assert!($indicator.$add(usize::MAX).is_err());
                            assert!(!$indicator.is_ready());
                            if active == $windows {
                                assert!($indicator.$add(1).is_err());
                            }

                            $update

                            assert_eq!($indicator.is_ready(), active > 0);
                            for index in 0..active {
                                assert!($indicator.is_ready_at(index));
                                assert_eq!($indicator.$read(index), Some($expected));
                            }
                            for index in [active, $windows, usize::MAX] {
                                assert!(!$indicator.is_ready_at(index));
                                assert_eq!($indicator.$read(index), None);
                            }
                            let before = $indicator.$read(0);
                            assert!($indicator.$add(1).is_err());
                            assert_eq!($indicator.$read(0), before);
                        })+
                    }
                }
            }
            check::<0>();
            check::<2>();
        }
    };
}

window_cases!(
    sample_sma_window_boundaries,
    |WINDOWS, warmup, sma| [
        SimpleMovingAverage::<StackRingBuffer<4, f64>, WINDOWS>::new_stack(warmup),
        SimpleMovingAverage::<HeapRingBuffer<f64>, WINDOWS>::new_heap(4, warmup),
    ],
    add_window,
    value_at,
    {
        for value in 1..=8 {
            sma.update(f64::from(value));
        }
        for (index, value) in sma.values().into_iter().enumerate() {
            assert_eq!(value.is_nan(), sma.value_at(index).is_none());
        }
        assert_eq!(
            sma.to_string().lines().count(),
            1 + sma.values().iter().filter(|v| !v.is_nan()).count()
        );
    },
    8.0
);

window_cases!(
    ema_window_boundaries,
    |WINDOWS, warmup, ema| [ExponentialMovingAverage::<WINDOWS>::new(warmup),],
    add_window,
    value_at,
    {
        ema.update(10.0);
        ema.update(20.0);
        for (index, value) in ema.values().into_iter().enumerate() {
            assert_eq!(value.is_nan(), ema.value_at(index).is_none());
        }
        assert_eq!(
            ema.to_string().lines().count(),
            1 + ema.values().iter().filter(|v| !v.is_nan()).count()
        );
    },
    20.0
);

window_cases!(
    cvd_window_boundaries,
    |WINDOWS, warmup, cvd| [
        CumulativeVolumeDelta::<StackRingBuffer<4, f64>, WINDOWS>::new_stack(warmup),
        CumulativeVolumeDelta::<HeapRingBuffer<f64>, WINDOWS>::new_heap(4, warmup),
    ],
    add_window,
    value_at,
    {
        for volume in 1..=8 {
            cvd.update(f64::from(volume), TradeSide::AggressorBuy)
                .unwrap();
        }
        cvd.update(3.0, TradeSide::AggressorSell).unwrap();
        for (index, value) in cvd.values().into_iter().enumerate() {
            assert_eq!(value.is_nan(), cvd.value_at(index).is_none());
        }
    },
    -3.0
);

window_cases!(
    sample_volatility_window_boundaries,
    |WINDOWS, warmup, volatility| [
        RollingVolatility::<StackRingBuffer<4, f64>, WINDOWS>::new_stack(warmup),
        RollingVolatility::<HeapRingBuffer<f64>, WINDOWS>::new_heap(4, warmup),
    ],
    add_window,
    value_at,
    {
        for _ in 0..8 {
            volatility.update(10.0);
        }
        for (index, value) in volatility.values().into_iter().enumerate() {
            assert_eq!(value.is_nan(), volatility.value_at(index).is_none());
        }
    },
    0.0
);

window_cases!(
    timed_sma_window_boundaries,
    |WINDOWS, warmup, sma| [
        SimpleMovingAverageTimed::<StackRingBuffer<4, (i64, f64)>, WINDOWS>::new_stack(
            Duration::from_secs(1),
            warmup
        )
        .unwrap(),
        SimpleMovingAverageTimed::<HeapRingBuffer<(i64, f64)>, WINDOWS>::new_heap(
            Duration::from_secs(1),
            4,
            warmup
        )
        .unwrap(),
    ],
    add_window_with_periods,
    value_at,
    {
        for timestamp in (0..8_000).step_by(1_000) {
            sma.update(10.0, timestamp);
            sma.update(20.0, timestamp);
        }
        sma.observe(10_000);
        sma.update(30.0, 10_000);
        for (index, value) in sma.values().into_iter().enumerate() {
            assert_eq!(value.is_nan(), sma.value_at(index).is_none());
        }
    },
    30.0
);

window_cases!(
    obv_window_boundaries,
    |WINDOWS, warmup, obv| [
        OnBalanceVolumeTimed::<StackRingBuffer<4, ObvBucket>, WINDOWS>::new_stack(
            Duration::from_secs(1),
            warmup
        )
        .unwrap(),
        OnBalanceVolumeTimed::<HeapRingBuffer<ObvBucket>, WINDOWS>::new_heap(
            Duration::from_secs(1),
            4,
            warmup
        )
        .unwrap(),
    ],
    add_window_with_periods,
    window_value,
    {
        for timestamp in (0..8_000).step_by(1_000) {
            obv.update(timestamp as f64, 2.0, timestamp);
        }
        obv.observe(10_000);
        obv.update(20_000.0, 3.0, 10_000);
    },
    3.0
);

window_cases!(
    timed_volatility_window_boundaries,
    |WINDOWS, warmup, volatility| [
        RollingVolatilityTimed::<StackRingBuffer<4, VolatilityBucket>, WINDOWS>::new_stack(
            Duration::from_secs(1),
            warmup
        )
        .unwrap(),
        RollingVolatilityTimed::<HeapRingBuffer<VolatilityBucket>, WINDOWS>::new_heap(
            Duration::from_secs(1),
            4,
            warmup
        )
        .unwrap(),
    ],
    add_window_with_periods,
    value_at,
    {
        for timestamp in (0..8_000).step_by(1_000) {
            volatility.update(10.0, timestamp);
            volatility.update(10.0, timestamp);
        }
        volatility.observe(10_000);
        volatility.update(10.0, 10_000);
        for (index, value) in volatility.values().into_iter().enumerate() {
            assert_eq!(value.is_nan(), volatility.value_at(index).is_none());
        }
    },
    0.0
);

window_cases!(
    timed_sum_volume_window_boundaries,
    |WINDOWS, warmup, volume| [
        RollingTradeVolumeTimed::<StackRingBuffer<4, TradeVolumeBucket>, WINDOWS>::new_stack(
            Duration::from_secs(1),
            warmup
        )
        .unwrap(),
        RollingTradeVolumeTimed::<HeapRingBuffer<TradeVolumeBucket>, WINDOWS>::new_heap(
            Duration::from_secs(1),
            4,
            warmup
        )
        .unwrap(),
    ],
    add_window_with_periods,
    window_value,
    {
        for timestamp in (0..8_000).step_by(1_000) {
            volume.update(2.0, timestamp);
            volume.update(3.0, timestamp);
        }
        volume.observe(10_000);
        volume.update(7.0, 10_000);
    },
    7.0
);

window_cases!(
    timed_sum_vwap_window_boundaries,
    |WINDOWS, warmup, vwap| [
        VolumeWeightedAveragePriceTimed::<StackRingBuffer<4, VwapBucket>, WINDOWS>::new_stack(
            Duration::from_secs(1),
            warmup
        )
        .unwrap(),
        VolumeWeightedAveragePriceTimed::<HeapRingBuffer<VwapBucket>, WINDOWS>::new_heap(
            Duration::from_secs(1),
            4,
            warmup
        )
        .unwrap(),
    ],
    add_window_with_periods,
    window_value,
    {
        for timestamp in (0..8_000).step_by(1_000) {
            vwap.update(10.0, 2.0, timestamp);
            vwap.update(20.0, 3.0, timestamp);
        }
        vwap.observe(10_000);
        vwap.update(30.0, 7.0, 10_000);
    },
    30.0
);

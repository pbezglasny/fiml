//! Volume-derived indicators, including CVD, OBV, VPT, rolling volume, and VWAP.
//!
mod cvd;
mod obv;
mod trade_volume;
mod vpt;
mod vwap;

pub use cvd::{CumulativeVolumeDelta, cvd};
pub use obv::{ObvBucket, OnBalanceVolumeTimed, obv_timed};
pub use trade_volume::{RollingTradeVolumeTimed, TradeVolumeBucket};
pub use vpt::{VolumePriceTrend, vpt};
pub use vwap::{VolumeWeightedAveragePriceTimed, VwapBucket};

use std::time::Duration;

use crate::{FimlError, HeapRingBuffer, IndicatorKind, InvalidArgumentError, Result, WarmupPolicy};

/// Calculates final rolling trade volume over one time-bucketed window.
pub fn trade_volume_timed(
    trades: &[(i64, f64)],
    window_periods: usize,
    aggregation: Duration,
    warmup_policy: WarmupPolicy,
) -> Result<Option<f64>> {
    let capacity = window_periods
        .checked_add(1)
        .ok_or(FimlError::InvalidArgument(
            InvalidArgumentError::TimedPeriodTooLarge {
                indicator: IndicatorKind::TradeVolumeTimed,
            },
        ))?;
    let mut calculator: RollingTradeVolumeTimed<HeapRingBuffer<TradeVolumeBucket>, 1> =
        RollingTradeVolumeTimed::new_heap(aggregation, capacity, warmup_policy)?;
    calculator.add_window_with_periods(window_periods)?;
    for &(timestamp, volume) in trades {
        calculator.update_inner(volume, timestamp);
    }
    Ok(calculator.window_value(0))
}

/// Calculates final VWAP over one time-bucketed window.
pub fn vwap_timed(
    trades: &[(i64, f64, f64)],
    window_periods: usize,
    aggregation: Duration,
    warmup_policy: WarmupPolicy,
) -> Result<Option<f64>> {
    let capacity = window_periods
        .checked_add(1)
        .ok_or(FimlError::InvalidArgument(
            InvalidArgumentError::TimedPeriodTooLarge {
                indicator: IndicatorKind::VwapTimed,
            },
        ))?;
    let mut calculator: VolumeWeightedAveragePriceTimed<HeapRingBuffer<VwapBucket>, 1> =
        VolumeWeightedAveragePriceTimed::new_heap(aggregation, capacity, warmup_policy)?;
    calculator.add_window_with_periods(window_periods)?;
    for &(timestamp, price, volume) in trades {
        calculator.update_inner(price, volume, timestamp);
    }
    Ok(calculator.window_value(0))
}

#[cfg(test)]
mod trade_volume_tests {
    use super::*;

    #[test]
    fn trade_volume_timed_returns_final_rolling_sum() {
        let result = trade_volume_timed(
            &[(0, 1.0), (1_000, 2.0), (2_000, 4.0)],
            2,
            Duration::from_secs(1),
            WarmupPolicy::FullWindow,
        );
        assert_eq!(result.unwrap(), Some(6.0));
    }
}

#[cfg(test)]
mod vwap_tests {
    use super::*;

    #[test]
    fn vwap_timed_returns_final_weighted_average() {
        let result = vwap_timed(
            &[(0, 10.0, 1.0), (1_000, 20.0, 3.0), (2_000, 30.0, 2.0)],
            2,
            Duration::from_secs(1),
            WarmupPolicy::FullWindow,
        );
        assert_eq!(result.unwrap(), Some(24.0));
    }
}

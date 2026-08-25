//! Timed on-balance volume indicator and final-value convenience calculation.

mod indicator;

pub use indicator::{ObvBucket, OnBalanceVolumeTimed};

use std::time::Duration;

use crate::{FimlError, Float, HeapRingBuffer, Result, WarmupPolicy};

/// Calculates the final time-bucketed on-balance volume over a rolling window.
///
/// # Arguments
///
/// * `trades` - `(timestamp, price, volume)` observations in chronological order. Timestamps are
///   expressed in milliseconds.
/// * `window_periods` - Number of aggregation buckets included in the rolling window. It must be
///   greater than zero.
/// * `aggregation` - Duration of each aggregation bucket. It must use whole-millisecond precision
///   and be at least one millisecond.
/// * `warmup_policy` - Determines whether a value is returned before the full window duration has
///   elapsed.
///
/// Returns `Ok(None)` when the input is empty or the configured warm-up policy has not been
/// satisfied.
///
/// # Errors
///
/// Returns [`FimlError::InvalidArgument`] when the aggregation or window length is invalid.
pub fn obv_timed<F: Float>(
    trades: &[(i64, F, F)],
    window_periods: usize,
    aggregation: Duration,
    warmup_policy: WarmupPolicy,
) -> Result<Option<F>> {
    let capacity = window_periods
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("OBV timed period is too large".to_string()))?;
    let mut calculator: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, 1> =
        OnBalanceVolumeTimed::new_heap(aggregation, capacity, warmup_policy)?;
    calculator.add_window_with_periods(window_periods)?;
    for &(timestamp, price, volume) in trades {
        calculator.update_inner(price, volume, timestamp);
    }
    Ok(calculator.window_value(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obv_timed_returns_final_rolling_value() {
        let trades = [(0, 100.0, 10.0), (1_000, 101.0, 7.0), (2_000, 99.0, 3.0)];

        let result = obv_timed(&trades, 2, Duration::from_secs(1), WarmupPolicy::FullWindow);

        assert_eq!(result.unwrap(), Some(4.0));
    }

    #[test]
    fn obv_timed_rejects_zero_window_periods() {
        assert!(
            obv_timed(
                &[(0, 100.0, 1.0)],
                0,
                Duration::from_secs(1),
                WarmupPolicy::FirstValue,
            )
            .is_err()
        );
    }
}

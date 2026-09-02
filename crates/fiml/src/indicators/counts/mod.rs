//! Count-based indicators and final-value convenience calculations.

mod trade_count;

pub use trade_count::{CountBucket, TradeCountTimed};

use std::time::Duration;

use crate::{HeapRingBuffer, Result, WarmupPolicy};

/// Calculates the final rolling trade count from a slice of trade timestamps.
///
/// # Arguments
///
/// * `timestamps` - Trade timestamps in chronological order, expressed in milliseconds.
/// * `window` - Duration covered by the rolling count. It must be a multiple of `aggregation` and
///   cannot be shorter than it.
/// * `aggregation` - Duration of each count bucket. It must use whole-millisecond precision and be
///   at least one millisecond.
/// * `warmup_policy` - Determines whether a count is returned before the full window duration has
///   elapsed.
///
/// Returns `Ok(None)` when the input is empty or the configured warm-up policy has not been
/// satisfied.
///
/// # Errors
///
/// Returns an error when the aggregation or window duration is invalid.
pub fn trade_count_timed(
    timestamps: &[i64],
    window: Duration,
    aggregation: Duration,
    warmup_policy: WarmupPolicy,
) -> Result<Option<f64>> {
    let mut calculator: TradeCountTimed<HeapRingBuffer<CountBucket>> =
        TradeCountTimed::new_heap(aggregation, window, warmup_policy)?;
    for &timestamp in timestamps {
        calculator.update(timestamp);
    }
    Ok(calculator.window_value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_count_timed_returns_final_rolling_count() {
        let result: Result<Option<f64>> = trade_count_timed(
            &[0, 100, 1_000, 2_000],
            Duration::from_secs(2),
            Duration::from_secs(1),
            WarmupPolicy::FullWindow,
        );

        assert_eq!(result.unwrap(), Some(2.0));
    }

    #[test]
    fn trade_count_timed_propagates_invalid_duration() {
        let result: Result<Option<f64>> = trade_count_timed(
            &[0],
            Duration::from_secs(1),
            Duration::ZERO,
            WarmupPolicy::FirstValue,
        );

        assert!(result.is_err());
    }
}

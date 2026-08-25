//! Simple moving average indicators and convenience functions for calculating final values.

mod indicator;

use std::time::Duration;

pub use indicator::{SimpleMovingAverage, SimpleMovingAverageTimed};

use crate::{FimlError, Float, HeapRingBuffer, Result, WarmupPolicy};

/// Calculates the final simple moving average after processing a slice of values.
///
/// # Arguments
///
/// * `values` - Input values in observation order.
/// * `window_length` - Number of observations included in the moving window. It must be greater
///   than zero.
/// * `warmup_policy` - Determines whether a value is returned before the window is full.
///
/// Returns `Ok(None)` when the input is empty or the configured warm-up policy has not been
/// satisfied.
///
/// # Errors
///
/// Returns [`FimlError::InvalidArgument`] when `window_length` is zero.
pub fn sma<F: Float>(
    values: &[F],
    window_length: usize,
    warmup_policy: WarmupPolicy,
) -> Result<Option<F>> {
    if window_length == 0 {
        return Err(FimlError::InvalidArgument(
            "Window period must be greater than 0".to_string(),
        ));
    }
    let mut calculator: SimpleMovingAverage<HeapRingBuffer<F>, F, 1> =
        SimpleMovingAverage::new_heap(window_length, warmup_policy);
    calculator.add_window(window_length)?;
    for &value in values {
        calculator.update(value);
    }
    Ok(calculator.value_at(0))
}

/// Calculates the final time-based simple moving average after processing timestamped values.
///
/// # Arguments
///
/// * `values` - `(timestamp, value)` observations in chronological order. Timestamps are expressed
///   in milliseconds.
/// * `window_duration` - Duration covered by the moving window. It must be a multiple of
///   `aggregation` and cannot be shorter than it.
/// * `aggregation` - Duration of each bucket used to aggregate observations. It must use
///   whole-millisecond precision and be at least one millisecond.
/// * `capacity` - Number of aggregation buckets retained in the ring buffer. It must be greater
///   than the number of buckets in `window_duration`.
/// * `warmup_policy` - Determines whether a value is returned before the full duration has elapsed.
///
/// Returns `Ok(None)` when the input is empty or the configured warm-up policy has not been
/// satisfied.
///
/// # Errors
///
/// Returns [`FimlError::InvalidArgument`] when the aggregation, window duration, or capacity is
/// invalid.
pub fn sma_timed<F: Float>(
    values: &[(i64, F)],
    window_duration: Duration,
    aggregation: Duration,
    capacity: usize,
    warmup_policy: WarmupPolicy,
) -> Result<Option<F>> {
    let mut calculator: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, 1> =
        SimpleMovingAverageTimed::new_heap(aggregation, capacity, warmup_policy)?;

    calculator.add_window_with_duration(window_duration)?;
    for &(timestamp, value) in values {
        calculator.update(value, timestamp);
    }
    Ok(calculator.value_at(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sma_returns_final_window_average() {
        let result = sma(&[1.0, 2.0, 3.0, 4.0], 3, WarmupPolicy::FirstValue);

        assert_eq!(result.unwrap(), Some(3.0));
    }

    #[test]
    fn sma_respects_full_window_warmup() {
        let result = sma(&[1.0, 2.0], 3, WarmupPolicy::FullWindow);

        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn sma_timed_returns_final_window_average() {
        let values = [(0, 10.0), (1_000, 20.0), (2_000, 30.0)];

        let result = sma_timed(
            &values,
            Duration::from_secs(2),
            Duration::from_secs(1),
            3,
            WarmupPolicy::FullWindow,
        );

        assert_eq!(result.unwrap(), Some(25.0));
    }

    #[test]
    fn invalid_arguments_return_errors() {
        assert!(sma(&[1.0], 0, WarmupPolicy::FirstValue).is_err());
        assert!(
            sma_timed(
                &[(0, 1.0)],
                Duration::from_secs(1),
                Duration::ZERO,
                1,
                WarmupPolicy::FirstValue,
            )
            .is_err()
        );
    }
}

//! Exponential moving average indicator and final-value convenience calculation.

mod indicator;

pub use indicator::ExponentialMovingAverage;

use crate::{Result, WarmupPolicy};

/// Calculates the final exponential moving average after processing a slice of values.
///
/// # Arguments
///
/// * `values` - Input values in observation order.
/// * `period` - Number of observations used to calculate the smoothing multiplier. It must be
///   greater than zero.
/// * `warmup_policy` - Determines whether a value is returned before `period` observations have
///   been processed.
///
/// Returns `Ok(None)` when the input is empty or the configured warm-up policy has not been
/// satisfied.
///
/// # Errors
///
/// Returns an error when `period` is zero or too large.
pub fn ema(values: &[f64], period: usize, warmup_policy: WarmupPolicy) -> Result<Option<f64>> {
    let mut calculator = ExponentialMovingAverage::<1>::new(warmup_policy);
    calculator.add_window(period)?;
    for &value in values {
        calculator.update(value);
    }
    Ok(calculator.value_at(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_returns_final_value() {
        let result = ema(&[10.0, 20.0, 30.0], 3, WarmupPolicy::FullWindow);

        assert_eq!(result.unwrap(), Some(22.5));
    }

    #[test]
    fn ema_propagates_invalid_period() {
        assert!(ema(&[10.0], 0, WarmupPolicy::FirstValue).is_err());
    }
}

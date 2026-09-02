//! Cumulative volume delta indicator and final-value convenience calculation.

mod indicator;

pub use indicator::CumulativeVolumeDelta;

use crate::{FimlError, HeapRingBuffer, InvalidArgumentError, Result, TradeSide, WarmupPolicy};

/// Calculates the final cumulative volume delta over a rolling trade window.
///
/// # Arguments
///
/// * `trades` - `(volume, trade_side)` observations in trade order. Aggressor-buy volume is added
///   and aggressor-sell volume is subtracted.
/// * `window_length` - Number of trades included in the rolling window. It must be greater than
///   zero.
/// * `warmup_policy` - Determines whether a value is returned before the window is full.
///
/// Returns `Ok(None)` when the input is empty or the configured warm-up policy has not been
/// satisfied.
///
/// # Errors
///
/// Returns [`FimlError::InvalidArgument`] when `window_length` is zero.
pub fn cvd(
    trades: &[(f64, TradeSide)],
    window_length: usize,
    warmup_policy: WarmupPolicy,
) -> Result<Option<f64>> {
    if window_length == 0 {
        return Err(FimlError::InvalidArgument(
            InvalidArgumentError::WindowPeriodZero,
        ));
    }
    let mut calculator: CumulativeVolumeDelta<HeapRingBuffer<f64>, 1> =
        CumulativeVolumeDelta::new_heap(window_length, warmup_policy);
    calculator.add_window(window_length)?;
    for &(volume, trade_side) in trades {
        calculator.update(volume, trade_side)?;
    }
    Ok(calculator.value_at(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cvd_returns_final_rolling_delta() {
        let trades = [
            (10.0, TradeSide::AgressorBuy),
            (3.0, TradeSide::AgressorSell),
            (7.0, TradeSide::AgressorBuy),
        ];

        let result = cvd(&trades, 2, WarmupPolicy::FullWindow);

        assert_eq!(result.unwrap(), Some(4.0));
    }

    #[test]
    fn cvd_rejects_zero_window_length() {
        assert!(
            cvd(
                &[(1.0, TradeSide::AgressorBuy)],
                0,
                WarmupPolicy::FirstValue
            )
            .is_err()
        );
    }
}

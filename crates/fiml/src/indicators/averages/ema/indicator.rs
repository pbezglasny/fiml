use std::fmt::Display;
use std::mem::MaybeUninit;

use crate::{FimlError, IntegerTarget, InvalidArgumentError, Result, WarmupPolicy};

/// Represents a single Exponential Moving Average (EMA) window.
pub struct EmaWindow {
    period: usize,
    multiplier: f64,
    moving_avg: Option<f64>,
}

impl Display for EmaWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.moving_avg {
            Some(moving_avg) => write!(
                f,
                "EmaWindow(period: {}, multiplier: {}, moving_avg: {})",
                self.period, self.multiplier, moving_avg
            ),
            None => write!(
                f,
                "EmaWindow(period: {}, multiplier: {}, moving_avg: None)",
                self.period, self.multiplier
            ),
        }
    }
}

/// Multiple Exponential Moving Averages (EMA) implementation.
///
/// Number of windows is fixed at compile time, but they can be added dynamically until number of
/// windows is reached. Windows must be added before any data is added to the EMA.
pub struct ExponentialMovingAverage<const WINDOWS: usize> {
    windows: [MaybeUninit<EmaWindow>; WINDOWS],
    window_count: usize,
    update_count: usize,
    warmup_policy: WarmupPolicy,
}

impl<const WINDOWS: usize> Display for ExponentialMovingAverage<WINDOWS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EMA with {} windows:", self.window_count)?;
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_ref() };
            writeln!(f, "  {}", window)?;
        }
        Ok(())
    }
}

impl<const WINDOWS: usize> ExponentialMovingAverage<WINDOWS> {
    pub fn new(warmup_policy: WarmupPolicy) -> Self {
        Self {
            windows: [const { MaybeUninit::<EmaWindow>::uninit() }; WINDOWS],
            window_count: 0,
            update_count: 0,
            warmup_policy,
        }
    }

    pub fn add_window(&mut self, period: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowLimitReached { limit: WINDOWS },
            ));
        }
        if self.update_count > 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowAddedAfterData,
            ));
        }
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodZero,
            ));
        }

        let divisor = period.checked_add(1).ok_or(FimlError::InvalidArgument(
            InvalidArgumentError::WindowPeriodOutOfRange {
                target: IntegerTarget::Usize,
            },
        ))?;
        let multiplier = 2.0 / divisor as f64;
        self.windows[self.window_count].write(EmaWindow {
            period,
            multiplier,
            moving_avg: None,
        });
        self.window_count += 1;
        #[cfg(feature = "tracing")]
        tracing::debug!(
            indicator = "EMA",
            window_index = self.window_count - 1,
            window_count = self.window_count,
            window_capacity = WINDOWS,
            period,
            "added indicator window"
        );
        Ok(())
    }

    pub fn update(&mut self, value: f64) {
        self.update_count += 1;
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_mut() };
            window.moving_avg = Some(if let Some(moving_avg) = window.moving_avg {
                let retained = 1.0 - window.multiplier;
                value * window.multiplier + moving_avg * retained
            } else {
                value
            });
        }
    }

    pub fn value_at(&self, index: usize) -> Option<f64> {
        if !self.is_ready_at(index) {
            return None;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        window.moving_avg
    }

    pub fn is_ready_at(&self, index: usize) -> bool {
        if index >= self.window_count {
            return false;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        match self.warmup_policy {
            WarmupPolicy::FirstValue => self.update_count > 0,
            WarmupPolicy::FullWindow => self.update_count >= window.period,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.window_count > 0 && (0..self.window_count).all(|index| self.is_ready_at(index))
    }

    pub fn values(&self) -> [f64; WINDOWS] {
        let mut result = [f64::NAN; WINDOWS];
        for (i, item) in result.iter_mut().enumerate().take(self.window_count) {
            if let Some(value) = self.value_at(i) {
                *item = value;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn exponential_moving_average_updates() {
        let mut ema: ExponentialMovingAverage<2> =
            ExponentialMovingAverage::new(WarmupPolicy::FirstValue);
        ema.add_window(3).unwrap();
        ema.add_window(5).unwrap();

        ema.update(10.0);
        assert!(approx_eq(ema.value_at(0).unwrap(), 10.0));
        assert!(approx_eq(ema.value_at(1).unwrap(), 10.0));

        ema.update(20.0);
        assert!(approx_eq(ema.value_at(0).unwrap(), 15.0));
        assert!(approx_eq(ema.value_at(1).unwrap(), 13.333333333333332));

        ema.update(30.0);
        assert!(approx_eq(ema.value_at(0).unwrap(), 22.5));
        assert!(approx_eq(ema.value_at(1).unwrap(), 18.888888888888886));
    }

    #[test]
    fn full_window_policy_withholds_ema_until_its_period_is_observed() {
        let mut ema: ExponentialMovingAverage<2> =
            ExponentialMovingAverage::new(WarmupPolicy::FullWindow);
        ema.add_window(2).unwrap();
        ema.add_window(3).unwrap();

        ema.update(10.0);
        assert_eq!(ema.value_at(0), None);
        assert_eq!(ema.value_at(1), None);

        ema.update(20.0);
        assert!(ema.is_ready_at(0));
        assert!(!ema.is_ready_at(1));
        assert!(ema.value_at(0).is_some());

        ema.update(30.0);
        assert!(ema.is_ready());
        assert!(ema.value_at(1).is_some());
    }

    #[test]
    fn exponential_moving_average_rejects_invalid_windows() {
        let mut ema: ExponentialMovingAverage<1> =
            ExponentialMovingAverage::new(WarmupPolicy::FirstValue);

        assert!(ema.add_window(0).is_err());
        assert!(ema.add_window(usize::MAX).is_err());
        ema.add_window(3).unwrap();
        assert!(ema.add_window(5).is_err());

        let mut updated_ema: ExponentialMovingAverage<2> =
            ExponentialMovingAverage::new(WarmupPolicy::FirstValue);
        updated_ema.add_window(3).unwrap();
        updated_ema.update(10.0);
        assert!(updated_ema.add_window(5).is_err());
    }
}

use std::fmt::Display;
use std::mem::MaybeUninit;

use crate::{FimlError, Float, Result};

/// Represents a single Exponential Moving Average (EMA) window.
pub struct EmaWindow<F: Float> {
    period: usize,
    multiplier: F,
    moving_avg: Option<F>,
}

impl<F> Display for EmaWindow<F>
where
    F: Float + Display,
{
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
pub struct ExponentialMovingAverage<T, const WINDOWS: usize>
where
    T: Float,
{
    windows: [MaybeUninit<EmaWindow<T>>; WINDOWS],
    window_count: usize,
    update_count: usize,
}

impl<T, const WINDOWS: usize> Display for ExponentialMovingAverage<T, WINDOWS>
where
    T: Float + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EMA with {} windows:", self.window_count)?;
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_ref() };
            writeln!(f, "  {}", window)?;
        }
        Ok(())
    }
}

impl<T, const WINDOWS: usize> ExponentialMovingAverage<T, WINDOWS>
where
    T: Float,
{
    pub fn new() -> Self {
        Self {
            windows: [const { MaybeUninit::<EmaWindow<T>>::uninit() }; WINDOWS],
            window_count: 0,
            update_count: 0,
        }
    }

    pub fn add_window(&mut self, period: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                "Maximum number of windows reached".to_string(),
            ));
        }
        if self.update_count > 0 {
            return Err(FimlError::InvalidArgument(
                "Cannot add window after data has been added".to_string(),
            ));
        }
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                "Window period must be greater than 0".to_string(),
            ));
        }

        let divisor = period
            .checked_add(1)
            .ok_or_else(|| FimlError::InvalidArgument("Window period is too large".to_string()))?;
        let multiplier = T::from_usize(2).div(T::from_usize(divisor));
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

    pub fn update(&mut self, value: T) {
        self.update_count += 1;
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_mut() };
            window.moving_avg = Some(if let Some(moving_avg) = window.moving_avg {
                let retained = T::ONE.sub(window.multiplier);
                value.mul(window.multiplier).add(moving_avg.mul(retained))
            } else {
                value
            });
        }
    }

    pub fn value_at(&self, index: usize) -> Option<T> {
        if index >= self.window_count {
            return None;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        window.moving_avg
    }

    pub fn values(&self) -> [T; WINDOWS] {
        let mut result = [T::ZERO; WINDOWS];
        for (i, item) in result.iter_mut().enumerate().take(self.window_count) {
            let window = unsafe { self.windows[i].assume_init_ref() };
            *item = window.moving_avg.unwrap_or(T::ZERO);
        }
        result
    }
}

impl<T, const WINDOWS: usize> Default for ExponentialMovingAverage<T, WINDOWS>
where
    T: Float,
{
    fn default() -> Self {
        Self::new()
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
        let mut ema: ExponentialMovingAverage<f64, 2> = ExponentialMovingAverage::new();
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
    fn exponential_moving_average_rejects_invalid_windows() {
        let mut ema: ExponentialMovingAverage<f64, 1> = ExponentialMovingAverage::new();

        assert!(ema.add_window(0).is_err());
        assert!(ema.add_window(usize::MAX).is_err());
        ema.add_window(3).unwrap();
        assert!(ema.add_window(5).is_err());

        let mut updated_ema: ExponentialMovingAverage<f64, 2> = ExponentialMovingAverage::new();
        updated_ema.add_window(3).unwrap();
        updated_ema.update(10.0);
        assert!(updated_ema.add_window(5).is_err());
    }
}

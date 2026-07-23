use std::mem::MaybeUninit;

use crate::features::TradeSide;
use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use crate::{FimlError, Float, Result};

struct CvdWindow<F: Float> {
    period: usize,
    current_value: F,
}

/// Cumulative volume delta (CVD) over one or more rolling trade windows.
///
/// Aggressor-buy volume is positive and aggressor-sell volume is negative.
/// Each configured window contains exactly its latest `period` trade deltas.
pub struct CumulativeVolumeDelta<R, F, const WINDOWS: usize>
where
    R: RingBuffer<Item = F>,
    F: Float,
{
    data: R,
    windows: [MaybeUninit<CvdWindow<F>>; WINDOWS],
    window_count: usize,
}

impl<const N: usize, F, const WINDOWS: usize>
    CumulativeVolumeDelta<StackRingBuffer<N, F>, F, WINDOWS>
where
    F: Float,
{
    pub fn new_stack() -> Self {
        Self::new(
            new_stack_ring_buffer(),
            [const { MaybeUninit::<CvdWindow<F>>::uninit() }; WINDOWS],
        )
    }
}

impl<F, const WINDOWS: usize> CumulativeVolumeDelta<HeapRingBuffer<F>, F, WINDOWS>
where
    F: Float,
{
    pub fn new_heap(periods: usize) -> Self {
        Self::new(
            new_heap_ring_buffer(periods),
            [const { MaybeUninit::<CvdWindow<F>>::uninit() }; WINDOWS],
        )
    }
}

impl<R, F, const WINDOWS: usize> CumulativeVolumeDelta<R, F, WINDOWS>
where
    R: RingBuffer<Item = F>,
    F: Float,
{
    fn new(data: R, windows: [MaybeUninit<CvdWindow<F>>; WINDOWS]) -> Self {
        Self {
            data,
            windows,
            window_count: 0,
        }
    }

    pub fn add_window(&mut self, period: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                "Maximum number of windows reached".to_string(),
            ));
        }
        if !self.data.is_empty() {
            return Err(FimlError::InvalidArgument(
                "Cannot add window after data has been added".to_string(),
            ));
        }
        if period > self.data.capacity() {
            return Err(FimlError::InvalidArgument(
                "Window period cannot be greater than ring buffer capacity".to_string(),
            ));
        }
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                "Window period must be greater than 0".to_string(),
            ));
        }

        self.windows[self.window_count].write(CvdWindow {
            period,
            current_value: F::ZERO,
        });

        self.window_count += 1;
        Ok(())
    }

    pub fn update(&mut self, volume: F, trade_side: TradeSide) -> Result<()> {
        self.update_inner(volume, trade_side);
        Ok(())
    }

    pub(crate) fn update_inner(&mut self, volume: F, trade_side: TradeSide) {
        let delta = match trade_side {
            TradeSide::AgressorBuy => volume,
            TradeSide::AgressorSell => -volume,
        };
        for window in 0..self.window_count {
            let window = unsafe { self.windows[window].assume_init_mut() };
            window.current_value += delta;
            if let Some(old_value) = self.data.peek_back_at(window.period - 1) {
                window.current_value -= *old_value;
            }
        }
        self.data.push_back(delta);
    }

    pub fn value_at(&self, index: usize) -> Option<F> {
        if index >= self.window_count {
            return None;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        Some(window.current_value)
    }

    pub fn values(&self) -> [F; WINDOWS] {
        let mut result = [F::ZERO; WINDOWS];
        for (index, value) in result.iter_mut().enumerate().take(self.window_count) {
            *value = self.value_at(index).unwrap();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_cvd<const CAPACITY: usize, const WINDOWS: usize>()
    -> CumulativeVolumeDelta<StackRingBuffer<CAPACITY, f64>, f64, WINDOWS> {
        CumulativeVolumeDelta::new_stack()
    }

    #[test]
    fn aggressive_buys_add_and_aggressive_sells_subtract_volume() {
        let mut cvd = new_cvd::<4, 1>();
        cvd.add_window(4).unwrap();

        cvd.update(10.0, TradeSide::AgressorBuy).unwrap();
        assert_eq!(cvd.value_at(0), Some(10.0));

        cvd.update(3.0, TradeSide::AgressorSell).unwrap();
        assert_eq!(cvd.value_at(0), Some(7.0));
    }

    #[test]
    fn rolling_window_keeps_exact_period() {
        let mut cvd = new_cvd::<3, 1>();
        cvd.add_window(2).unwrap();

        cvd.update(10.0, TradeSide::AgressorBuy).unwrap();
        cvd.update(3.0, TradeSide::AgressorSell).unwrap();
        assert_eq!(cvd.value_at(0), Some(7.0));

        cvd.update(7.0, TradeSide::AgressorBuy).unwrap();
        assert_eq!(cvd.value_at(0), Some(4.0));
    }

    #[test]
    fn window_equal_to_buffer_capacity_slides() {
        let mut cvd = new_cvd::<3, 1>();
        cvd.add_window(3).unwrap();

        for volume in [1.0, 2.0, 3.0] {
            cvd.update(volume, TradeSide::AgressorBuy).unwrap();
        }
        assert_eq!(cvd.value_at(0), Some(6.0));

        cvd.update(4.0, TradeSide::AgressorBuy).unwrap();
        assert_eq!(cvd.value_at(0), Some(9.0));
    }

    #[test]
    fn multiple_windows_are_independent() {
        let mut cvd = new_cvd::<4, 2>();
        cvd.add_window(2).unwrap();
        cvd.add_window(4).unwrap();

        for (volume, side) in [
            (10.0, TradeSide::AgressorBuy),
            (3.0, TradeSide::AgressorSell),
            (7.0, TradeSide::AgressorBuy),
            (2.0, TradeSide::AgressorSell),
            (5.0, TradeSide::AgressorBuy),
        ] {
            cvd.update(volume, side).unwrap();
        }

        assert_eq!(cvd.values(), [3.0, 7.0]);
    }

    #[test]
    fn add_window_errors_when_full() {
        let mut cvd = new_cvd::<3, 1>();
        cvd.add_window(1).unwrap();

        assert!(cvd.add_window(2).is_err());
    }

    #[test]
    fn add_window_errors_after_data() {
        let mut cvd = new_cvd::<3, 2>();
        cvd.add_window(1).unwrap();
        cvd.update(1.0, TradeSide::AgressorBuy).unwrap();

        assert!(cvd.add_window(2).is_err());
    }

    #[test]
    fn add_window_rejects_invalid_periods() {
        let mut cvd = new_cvd::<3, 1>();

        assert!(cvd.add_window(0).is_err());
        assert!(cvd.add_window(4).is_err());
        assert!(cvd.add_window(3).is_ok());
    }
}

use crate::FimlError;
use crate::Float;
use crate::Result;
use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use std::fmt::Display;
use std::mem::MaybeUninit;

/// Represents a single Simple Moving Average (SMA) window, which tracks the period
pub struct SmaWindow<F: Float> {
    period: usize,
    sum: F,
    moving_avg: F,
}

impl<F> Display for SmaWindow<F>
where
    F: Float + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SmaWindow(period: {}, sum: {}, moving_avg: {})",
            self.period, self.sum, self.moving_avg
        )
    }
}

/// Multiple Simple Moving Averages (SMA) implementation that can be used with both stack and heap
/// ring buffers.
/// Number of windows is fixed at compile time, but they can be added dynamically until number of
/// winndows is reached.
/// You can add windows with different periods, but they must be added before any data is added to
/// the SMA. Once data is added, you cannot add more windows.
pub struct SimpleMovingAverage<R, T, const WINDOWS: usize>
where
    R: RingBuffer<Item = T>,
    T: Float,
{
    data: R,
    windows: [MaybeUninit<SmaWindow<T>>; WINDOWS],
    window_count: usize,
}

impl<R, T, const WINDOWS: usize> Display for SimpleMovingAverage<R, T, WINDOWS>
where
    R: RingBuffer<Item = T>,
    T: Float + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SMA with {} windows:", self.window_count)?;
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_ref() };
            writeln!(f, "  {}", window)?;
        }
        Ok(())
    }
}

impl<R, T, const WINDOWS: usize> SimpleMovingAverage<R, T, WINDOWS>
where
    R: RingBuffer<Item = T>,
    T: Float,
{
    fn new(data: R, windows: [MaybeUninit<SmaWindow<T>>; WINDOWS]) -> Self {
        Self {
            data,
            windows,
            window_count: 0,
        }
    }
}

impl<const PERIODS: usize, T, const WINDOWS: usize>
    SimpleMovingAverage<StackRingBuffer<PERIODS, T>, T, WINDOWS>
where
    T: Float,
{
    pub fn new_stack() -> Self {
        let stack_data = new_stack_ring_buffer::<PERIODS, T>();
        let windows = [const { MaybeUninit::<SmaWindow<T>>::uninit() }; WINDOWS];
        Self::new(stack_data, windows)
    }
}

impl<T, const WINDOWS: usize> SimpleMovingAverage<HeapRingBuffer<T>, T, WINDOWS>
where
    T: Float,
{
    pub fn new_heap(periods: usize) -> Self {
        let heap_data = new_heap_ring_buffer::<T>(periods);
        let windows = [const { MaybeUninit::<SmaWindow<T>>::uninit() }; WINDOWS];
        Self::new(heap_data, windows)
    }
}

impl<R, T, const WINDOWS: usize> SimpleMovingAverage<R, T, WINDOWS>
where
    R: RingBuffer<Item = T>,
    T: Float,
{
    pub fn add_window(&mut self, period: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                "Maximum number of windows reached".to_string(),
            ));
        }
        if self.data.len() > 0 {
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
        self.windows[self.window_count].write(SmaWindow {
            period,
            sum: T::zero(),
            moving_avg: T::zero(),
        });
        self.window_count += 1;
        Ok(())
    }

    pub fn update(&mut self, value: T) {
        let old_value = self.data.push_back(value);
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_mut() };
            let prev_value = if window.period == self.data.capacity() {
                old_value.as_ref()
            } else if window.period < self.data.capacity() {
                self.data.peek_back_at(window.period)
            } else {
                None
            };

            window.sum = window
                .sum
                .add(value)
                .sub(prev_value.copied().unwrap_or(T::zero()));
            let divisor = if self.data.capacity() < window.period {
                T::from_usize(self.data.capacity())
            } else {
                T::from_usize(window.period)
            };
            window.moving_avg = window.sum.div(divisor);
        }
    }

    pub fn value_at(&self, index: usize) -> Option<T> {
        if index >= self.window_count {
            return None;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        Some(window.moving_avg)
    }

    pub fn values(&self) -> [T; WINDOWS] {
        let mut result = [T::zero(); WINDOWS];
        for (i, item) in result.iter_mut().enumerate().take(self.window_count) {
            let window = unsafe { self.windows[i].assume_init_ref() };
            *item = window.moving_avg;
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
    fn new_stack_has_no_windows() {
        let sma: SimpleMovingAverage<StackRingBuffer<4, f64>, f64, 2> =
            SimpleMovingAverage::new_stack();
        assert_eq!(sma.value_at(0), None);
        assert_eq!(sma.value_at(1), None);
    }

    #[test]
    fn add_window_increments_count() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<4, f64>, f64, 2> =
            SimpleMovingAverage::new_stack();
        assert!(sma.add_window(2).is_ok());
        assert!(sma.add_window(3).is_ok());
        assert_eq!(sma.value_at(0), Some(0.0));
        assert_eq!(sma.value_at(1), Some(0.0));
        assert_eq!(sma.value_at(2), None);
    }

    #[test]
    fn add_window_errors_when_full() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<4, f64>, f64, 2> =
            SimpleMovingAverage::new_stack();
        sma.add_window(2).unwrap();
        sma.add_window(3).unwrap();
        assert!(sma.add_window(4).is_err());
    }

    #[test]
    fn add_window_errors_after_data() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<4, f64>, f64, 2> =
            SimpleMovingAverage::new_stack();
        sma.add_window(2).unwrap();
        sma.update(1.0);
        assert!(sma.add_window(3).is_err());
    }

    #[test]
    fn moving_average_within_period() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<4, f64>, f64, 1> =
            SimpleMovingAverage::new_stack();
        sma.add_window(3).unwrap();

        sma.update(3.0);
        assert!(approx_eq(sma.value_at(0).unwrap(), 1.0));

        sma.update(6.0);
        assert!(approx_eq(sma.value_at(0).unwrap(), 3.0));

        sma.update(9.0);
        assert!(approx_eq(sma.value_at(0).unwrap(), 6.0));
    }

    #[test]
    fn moving_average_slides_at_capacity() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<3, f64>, f64, 1> =
            SimpleMovingAverage::new_stack();
        sma.add_window(3).unwrap();

        sma.update(1.0);
        sma.update(2.0);
        sma.update(3.0);
        assert!(approx_eq(sma.value_at(0).unwrap(), 2.0));

        sma.update(4.0);
        assert!(approx_eq(sma.value_at(0).unwrap(), 3.0));

        sma.update(5.0);
        assert!(approx_eq(sma.value_at(0).unwrap(), 4.0));
    }

    #[test]
    fn multiple_windows_independent() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<5, f64>, f64, 2> =
            SimpleMovingAverage::new_stack();
        sma.add_window(2).unwrap();
        sma.add_window(5).unwrap();

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            sma.update(v);
        }

        assert!(approx_eq(sma.value_at(0).unwrap(), 4.5));
        assert!(approx_eq(sma.value_at(1).unwrap(), 3.0));
    }

    #[test]
    fn values_returns_all_window_averages() {
        let mut sma: SimpleMovingAverage<StackRingBuffer<5, f64>, f64, 2> =
            SimpleMovingAverage::new_stack();
        sma.add_window(2).unwrap();
        sma.add_window(5).unwrap();

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            sma.update(v);
        }

        let values = sma.values();
        assert!(approx_eq(values[0], 4.5));
        assert!(approx_eq(values[1], 3.0));
    }

    #[test]
    fn heap_variant_matches_stack() {
        let mut stack_sma: SimpleMovingAverage<StackRingBuffer<4, f64>, f64, 1> =
            SimpleMovingAverage::new_stack();
        let mut heap_sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, 1> =
            SimpleMovingAverage::new_heap(4);
        stack_sma.add_window(3).unwrap();
        heap_sma.add_window(3).unwrap();

        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            stack_sma.update(v);
            heap_sma.update(v);
            assert!(approx_eq(
                stack_sma.value_at(0).unwrap(),
                heap_sma.value_at(0).unwrap(),
            ));
        }
    }
}

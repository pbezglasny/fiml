use std::fmt::Display;
use std::mem::MaybeUninit;
use std::time::Duration;

use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use crate::{FimlError, Float, Result};

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
            sum: T::ZERO,
            moving_avg: T::ZERO,
        });
        self.window_count += 1;
        #[cfg(feature = "tracing")]
        tracing::debug!(
            indicator = "SMA",
            window_index = self.window_count - 1,
            window_count = self.window_count,
            window_capacity = WINDOWS,
            period,
            "added indicator window"
        );
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
                .sub(prev_value.copied().unwrap_or(T::ZERO));
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
        let mut result = [T::ZERO; WINDOWS];
        for (i, item) in result.iter_mut().enumerate().take(self.window_count) {
            let window = unsafe { self.windows[i].assume_init_ref() };
            *item = window.moving_avg;
        }
        result
    }
}

struct SmaWindowTimed<T: Float> {
    duration: i64,
    bucket_count: usize,
    sum: T,
    moving_avg: T,
}

///Simple Moving Average(SMA) with time-based windows. Event stream aggregated for specified duration.
///Each window tracks the sum and count of values within its time period, and calculates the average
///as sum/count.
///Window cannot be less than the aggregation duration, and all windows must be multiples of the
///aggregation duration.
///Aggeregation can not be less than one millisecond.
pub struct SimpleMovingAverageTimed<R, T, const WINDOWS: usize>
where
    R: RingBuffer<Item = (i64, T)>,
    T: Float,
{
    data: R,
    millis_aggregation: i64,
    last_sum: Option<T>,
    last_cnt: usize,
    windows: [MaybeUninit<SmaWindowTimed<T>>; WINDOWS],
    window_count: usize,
}

impl<const N: usize, T, const WINDOWS: usize>
    SimpleMovingAverageTimed<StackRingBuffer<N, (i64, T)>, T, WINDOWS>
where
    T: Float,
{
    ///Create new SimpleMovingAverageTimed with stack ring buffer.
    pub fn new_stack(aggeregation: Duration) -> Result<Self> {
        if N == 0 {
            return Err(FimlError::InvalidArgument(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }
        let stack_data = new_stack_ring_buffer::<N, (i64, T)>();
        Self::new_with_buffer(stack_data, aggeregation, N)
    }
}

impl<T, const WINDOWS: usize> SimpleMovingAverageTimed<HeapRingBuffer<(i64, T)>, T, WINDOWS>
where
    T: Float,
{
    /// Create new SimpleMovingAverageTimed with heap ring buffer.
    pub fn new_heap(aggeregation: Duration, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }
        let heap_data = new_heap_ring_buffer::<(i64, T)>(capacity);
        Self::new_with_buffer(heap_data, aggeregation, capacity)
    }
}

impl<R, T, const WINDOWS: usize> SimpleMovingAverageTimed<R, T, WINDOWS>
where
    R: RingBuffer<Item = (i64, T)>,
    T: Float,
{
    fn new_with_buffer(data: R, aggeregation: Duration, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }
        if aggeregation.as_millis() == 0 {
            return Err(FimlError::InvalidArgument(
                "Aggregation duration must be at least 1 millisecond".to_string(),
            ));
        }
        let millis_aggregation = aggeregation.as_millis() as i64;
        Ok(Self {
            data,
            millis_aggregation,
            last_sum: None,
            last_cnt: 0,
            windows: [const { MaybeUninit::<SmaWindowTimed<T>>::uninit() }; WINDOWS],
            window_count: 0,
        })
    }

    pub fn add_window_with_periods(&mut self, periods: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                "Maximum number of windows reached".to_string(),
            ));
        }
        if periods == 0 {
            return Err(FimlError::InvalidArgument(
                "Window period must be greater than 0".to_string(),
            ));
        }
        if periods >= self.data.capacity() {
            return Err(FimlError::InvalidArgument(
                "Window period must be less than ring buffer capacity".to_string(),
            ));
        }
        self.windows[self.window_count].write(SmaWindowTimed {
            duration: periods as i64 * self.millis_aggregation,
            bucket_count: 0,
            sum: T::ZERO,
            moving_avg: T::ZERO,
        });
        self.window_count += 1;
        #[cfg(feature = "tracing")]
        tracing::debug!(
            indicator = "SMA timed",
            window_index = self.window_count - 1,
            window_count = self.window_count,
            window_capacity = WINDOWS,
            periods,
            duration_millis = periods as i64 * self.millis_aggregation,
            "added indicator window"
        );
        Ok(())
    }

    pub fn add_window_with_duration(&mut self, period: Duration) -> Result<()> {
        let millis_period = period.as_millis() as i64;
        if millis_period < self.millis_aggregation {
            return Err(FimlError::InvalidArgument(
                "Window period cannot be less than aggregation duration".to_string(),
            ));
        }
        if millis_period % self.millis_aggregation != 0 {
            return Err(FimlError::InvalidArgument(
                "Window period must be a multiple of aggregation duration".to_string(),
            ));
        }
        let period_in_aggregations = (millis_period / self.millis_aggregation) as usize;
        self.add_window_with_periods(period_in_aggregations)
    }

    fn expire_old_buckets(data: &R, window: &mut SmaWindowTimed<T>, now: i64) {
        while window.bucket_count > 0 {
            let index = data.len() - window.bucket_count;
            let (date, value) = data.peek_front_at(index).unwrap();
            if *date + window.duration > now {
                break;
            }
            window.sum = window.sum.sub(*value);
            window.bucket_count -= 1;
        }
        window.moving_avg = if window.bucket_count > 0 {
            window.sum.div(T::from_usize(window.bucket_count))
        } else {
            T::ZERO
        };
    }

    #[inline]
    fn update_moving_avg(window: &mut SmaWindowTimed<T>) {
        window.moving_avg = if window.bucket_count > 0 {
            window.sum.div(T::from_usize(window.bucket_count))
        } else {
            T::ZERO
        };
    }

    fn bucket_start(&self, timestamp: i64) -> i64 {
        timestamp - timestamp.rem_euclid(self.millis_aggregation)
    }

    pub(crate) fn update_inner(&mut self, value: T, now: i64) {
        let bucket_start = self.bucket_start(now);
        for i in 0..self.window_count {
            let window = unsafe { self.windows[i].assume_init_mut() };
            Self::expire_old_buckets(&self.data, window, now);
        }

        let update_last_bucket = self
            .data
            .peek_back()
            .is_some_and(|(date, _)| *date == bucket_start);

        // We are in the same aggregation bucket as the last update. Update the last value
        if update_last_bucket {
            self.last_sum = self.last_sum.or(Some(T::ZERO)).map(|v| v.add(value));
            self.last_cnt += 1;
            let (prev_date, prev_value) = self.data.pop_back().unwrap();
            let new_value = self.last_sum.unwrap().div(T::from_usize(self.last_cnt));
            self.data.push_back((prev_date, new_value));
            for i in 0..self.window_count {
                let window = unsafe { self.windows[i].assume_init_mut() };
                if window.bucket_count > 0 {
                    window.sum = window.sum.sub(prev_value).add(new_value);
                } else if prev_date + window.duration > now {
                    window.sum = window.sum.add(new_value);
                    window.bucket_count = 1;
                }
                Self::update_moving_avg(window);
            }
        } else {
            self.last_sum = Some(value);
            self.last_cnt = 1;
            let len_before = self.data.len();
            let old_value = self.data.push_back((bucket_start, value));
            for i in 0..self.window_count {
                let window = unsafe { self.windows[i].assume_init_mut() };
                if let Some((_, old_value)) = old_value.as_ref()
                    && window.bucket_count == len_before
                {
                    window.sum = window.sum.sub(*old_value);
                    window.bucket_count -= 1;
                }
                if window.duration > 0 {
                    window.sum = window.sum.add(value);
                    window.bucket_count += 1;
                }
                Self::update_moving_avg(window);
            }
        }
    }

    pub fn update(&mut self, value: T) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as i64;
        self.update_inner(value, now);
    }

    /// Return value of i-th Window
    pub fn value_at(&self, index: usize) -> Option<T> {
        if index >= self.window_count {
            return None;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        Some(window.moving_avg)
    }

    /// Return values of windows
    pub fn values(&self) -> [T; WINDOWS] {
        let mut result = [T::ZERO; WINDOWS];
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

    fn timed_value_at<R, const WINDOWS: usize>(
        sma: &SimpleMovingAverageTimed<R, f64, WINDOWS>,
        index: usize,
    ) -> f64
    where
        R: RingBuffer<Item = (i64, f64)>,
    {
        let window = unsafe { sma.windows[index].assume_init_ref() };
        window.moving_avg
    }

    fn timed_bucket_count<R, const WINDOWS: usize>(
        sma: &SimpleMovingAverageTimed<R, f64, WINDOWS>,
        index: usize,
    ) -> usize
    where
        R: RingBuffer<Item = (i64, f64)>,
    {
        let window = unsafe { sma.windows[index].assume_init_ref() };
        window.bucket_count
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

    #[test]
    fn timed_moving_average_updates_before_buffer_is_full() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 8).unwrap();
        sma.add_window_with_duration(Duration::from_millis(3_000))
            .unwrap();

        sma.update_inner(1.0, 0);
        sma.update_inner(2.0, 1_000);

        assert!(approx_eq(timed_value_at(&sma, 0), 1.5));
    }

    #[test]
    fn timed_stack_variant_uses_const_capacity() {
        let mut sma: SimpleMovingAverageTimed<StackRingBuffer<3, (i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_stack(Duration::from_millis(1_000)).unwrap();
        sma.add_window_with_periods(2).unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 1_000);
        sma.update_inner(30.0, 2_000);

        assert_eq!(sma.data.capacity(), 3);
        assert_eq!(sma.data.len(), 3);
        assert!(approx_eq(timed_value_at(&sma, 0), 25.0));
    }

    #[test]
    fn timed_new_heap_rejects_zero_capacity() {
        let sma: Result<SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1>> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 0);

        assert!(sma.is_err());
    }

    #[test]
    fn timed_new_stack_rejects_zero_capacity() {
        let sma: Result<SimpleMovingAverageTimed<StackRingBuffer<0, (i64, f64)>, f64, 1>> =
            SimpleMovingAverageTimed::new_stack(Duration::from_millis(1_000));

        assert!(sma.is_err());
    }

    #[test]
    fn timed_add_window_with_duration_rejects_invalid_duration() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 2> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();

        assert!(
            sma.add_window_with_duration(Duration::from_millis(500))
                .is_err()
        );
        assert!(
            sma.add_window_with_duration(Duration::from_millis(1_500))
                .is_err()
        );
        assert!(
            sma.add_window_with_duration(Duration::from_millis(2_000))
                .is_ok()
        );
    }

    #[test]
    fn timed_add_window_with_periods_rejects_invalid_periods() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 2> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();

        assert!(sma.add_window_with_periods(0).is_err());
        assert!(sma.add_window_with_periods(3).is_err());
        assert!(sma.add_window_with_periods(4).is_err());
        assert!(sma.add_window_with_periods(2).is_ok());
    }

    #[test]
    fn timed_moving_average_uses_same_bucket_average() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        sma.add_window_with_duration(Duration::from_millis(1_000))
            .unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 500);

        assert!(approx_eq(timed_value_at(&sma, 0), 15.0));
    }

    #[test]
    fn timed_moving_average_aligns_buckets_to_aggregation_boundaries() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        sma.add_window_with_duration(Duration::from_millis(2_000))
            .unwrap();

        sma.update_inner(10.0, 999);
        sma.update_inner(20.0, 1_000);

        assert_eq!(sma.data.len(), 2);
        assert!(approx_eq(timed_value_at(&sma, 0), 15.0));
    }

    #[test]
    fn timed_same_bucket_update_can_readd_expired_bucket() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        sma.add_window_with_duration(Duration::from_millis(1_000))
            .unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 1_000);
        sma.update_inner(40.0, 1_500);

        assert_eq!(timed_bucket_count(&sma, 0), 1);
        assert!(approx_eq(timed_value_at(&sma, 0), 30.0));
    }

    #[test]
    fn timed_moving_average_expires_old_buckets() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        sma.add_window_with_duration(Duration::from_millis(2_000))
            .unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 1_000);
        sma.update_inner(30.0, 2_000);

        assert!(approx_eq(timed_value_at(&sma, 0), 25.0));
    }

    #[test]
    fn timed_multiple_windows_are_independent() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 2> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        sma.add_window_with_periods(1).unwrap();
        sma.add_window_with_periods(3).unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 1_000);
        sma.update_inner(30.0, 2_000);
        sma.update_inner(40.0, 2_500);
        sma.update_inner(50.0, 3_000);

        assert_eq!(timed_bucket_count(&sma, 0), 1);
        assert_eq!(timed_bucket_count(&sma, 1), 3);
        assert!(approx_eq(timed_value_at(&sma, 0), 50.0));
        assert!(approx_eq(timed_value_at(&sma, 1), 35.0));
    }

    #[test]
    fn timed_moving_average_handles_capacity_overwrite() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        sma.add_window_with_periods(2).unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 1_000);
        sma.update_inner(30.0, 2_000);
        sma.update_inner(40.0, 3_000);

        assert_eq!(sma.data.len(), 3);
        assert_eq!(timed_bucket_count(&sma, 0), 2);
        assert!(approx_eq(timed_value_at(&sma, 0), 35.0));
    }

    #[test]
    fn timed_moving_average_window_period_must_be_less_than_capacity() {
        let mut sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, f64, 1> =
            SimpleMovingAverageTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();

        assert!(sma.add_window_with_periods(3).is_err());
        sma.add_window_with_periods(2).unwrap();

        sma.update_inner(10.0, 0);
        sma.update_inner(20.0, 1_000);
        sma.update_inner(30.0, 2_000);

        assert_eq!(sma.data.capacity(), 3);
        assert_eq!(sma.data.len(), 3);
        assert!(approx_eq(timed_value_at(&sma, 0), 25.0));
    }
}

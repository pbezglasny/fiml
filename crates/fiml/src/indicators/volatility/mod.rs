//! Rolling population volatility over simple returns.

use std::mem::MaybeUninit;
use std::time::Duration;

use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use crate::{DurationField, FimlError, IntegerTarget, InvalidArgumentError, Result, WarmupPolicy};

#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn add_value(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    fn remove_value(&mut self, value: f64) {
        if self.count == 1 {
            *self = Self::default();
            return;
        }
        self.count -= 1;
        let delta = value - self.mean;
        self.mean -= delta / self.count as f64;
        self.m2 -= delta * (value - self.mean);
    }

    fn merge(&mut self, other: Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other;
            return;
        }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.m2 += other.m2
            + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean += delta * other.count as f64 / combined_count as f64;
        self.count = combined_count;
    }

    fn remove(&mut self, other: Self) {
        let remaining_count = self.count - other.count;
        if remaining_count == 0 {
            *self = Self::default();
            return;
        }
        let remaining_mean = (self.count as f64 * self.mean - other.count as f64 * other.mean)
            / remaining_count as f64;
        let delta = other.mean - remaining_mean;
        self.m2 -= other.m2
            + delta * delta * remaining_count as f64 * other.count as f64 / self.count as f64;
        self.mean = remaining_mean;
        self.count = remaining_count;
    }

    fn volatility(self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some((self.m2 / self.count as f64).max(0.0).sqrt())
    }
}

struct VolatilityWindow {
    period: usize,
    moments: Moments,
}

/// Rolling population standard deviation of simple returns over sample windows.
///
/// The first value establishes the return baseline. A zero previous value also
/// establishes a new baseline because its percentage return is undefined.
pub struct RollingVolatility<R, const WINDOWS: usize>
where
    R: RingBuffer<Item = f64>,
{
    data: R,
    windows: [MaybeUninit<VolatilityWindow>; WINDOWS],
    window_count: usize,
    warmup_policy: WarmupPolicy,
    previous_value: Option<f64>,
}

impl<R, const WINDOWS: usize> RollingVolatility<R, WINDOWS>
where
    R: RingBuffer<Item = f64>,
{
    fn new(data: R, warmup_policy: WarmupPolicy) -> Self {
        Self {
            data,
            windows: [const { MaybeUninit::uninit() }; WINDOWS],
            window_count: 0,
            warmup_policy,
            previous_value: None,
        }
    }

    /// Adds a window containing `period` simple returns.
    ///
    /// # Errors
    ///
    /// Rejects additions after data has arrived or once `WINDOWS` windows are configured. Also
    /// rejects zero periods or periods exceeding history capacity.
    pub fn add_window(&mut self, period: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowLimitReached { limit: WINDOWS },
            ));
        }
        if !self.data.is_empty() || self.previous_value.is_some() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowAddedAfterData,
            ));
        }
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodZero,
            ));
        }
        if period > self.data.capacity() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodExceedsCapacity {
                    period,
                    capacity: self.data.capacity(),
                },
            ));
        }
        self.windows[self.window_count].write(VolatilityWindow {
            period,
            moments: Moments::default(),
        });
        self.window_count += 1;
        Ok(())
    }

    /// Records a value and updates volatility using its simple return from the previous value.
    /// The first value, or a zero previous value, only establishes a new baseline.
    /// The caller must supply finite values.
    pub fn update(&mut self, value: f64) {
        let Some(previous_value) = self.previous_value.replace(value) else {
            return;
        };
        if previous_value == 0.0 {
            return;
        }
        let current_return = (value - previous_value) / previous_value;
        let evicted = self.data.push_back(current_return);
        for index in 0..self.window_count {
            let window = unsafe { self.windows[index].assume_init_mut() };
            let expired = if window.period == self.data.capacity() {
                evicted.as_ref()
            } else {
                self.data.peek_back_at(window.period)
            };
            window.moments.add_value(current_return);
            if let Some(&expired) = expired {
                window.moments.remove_value(expired);
            }
        }
    }

    /// Returns the zero-based window's value, or `None` if absent or still warming up.
    /// Also returns `None` when no return observations are available.
    pub fn value_at(&self, index: usize) -> Option<f64> {
        self.is_ready_at(index)
            .then(|| {
                unsafe { self.windows[index].assume_init_ref() }
                    .moments
                    .volatility()
            })
            .flatten()
    }

    /// Reports whether the zero-based window has met its warm-up policy; false if absent.
    /// Readiness does not guarantee a value when no returns are available.
    pub fn is_ready_at(&self, index: usize) -> bool {
        if index >= self.window_count {
            return false;
        }
        let window = unsafe { self.windows[index].assume_init_ref() };
        match self.warmup_policy {
            WarmupPolicy::FirstValue => self.previous_value.is_some(),
            WarmupPolicy::FullWindow => self.data.len() >= window.period,
        }
    }

    /// Returns true when at least one window exists and all windows meet their warm-up policy.
    pub fn is_ready(&self) -> bool {
        self.window_count > 0 && (0..self.window_count).all(|index| self.is_ready_at(index))
    }

    /// Returns window values in insertion order, with `NaN` for unused or unavailable slots.
    pub fn values(&self) -> [f64; WINDOWS] {
        let mut values = [f64::NAN; WINDOWS];
        for (index, value) in values.iter_mut().enumerate().take(self.window_count) {
            if let Some(volatility) = self.value_at(index) {
                *value = volatility;
            }
        }
        values
    }
}

impl<const PERIODS: usize, const WINDOWS: usize>
    RollingVolatility<StackRingBuffer<PERIODS, f64>, WINDOWS>
{
    /// Creates an empty indicator with inline history; add windows before updating.
    ///
    /// # Panics
    ///
    /// Panics if the compile-time history capacity is zero.
    pub fn new_stack(warmup_policy: WarmupPolicy) -> Self {
        Self::new(new_stack_ring_buffer(), warmup_policy)
    }
}

impl<const WINDOWS: usize> RollingVolatility<HeapRingBuffer<f64>, WINDOWS> {
    /// Creates an empty indicator with heap-allocated history; add windows before updating.
    ///
    /// # Panics
    ///
    /// Panics if `periods` is zero.
    pub fn new_heap(periods: usize, warmup_policy: WarmupPolicy) -> Self {
        Self::new(new_heap_ring_buffer(periods), warmup_policy)
    }
}

/// One fixed-duration bucket of simple-return moments.
#[derive(Clone, Copy)]
pub struct VolatilityBucket {
    timestamp: i64,
    moments: Moments,
}

struct TimedVolatilityWindow {
    duration: i64,
    bucket_count: usize,
    moments: Moments,
    ready: bool,
}

/// Rolling population standard deviation of simple returns over timed windows.
///
/// Returns within an aggregation bucket are represented by their exact count,
/// mean and central moment, so updates require fixed memory and no allocation.
pub struct RollingVolatilityTimed<R, const WINDOWS: usize>
where
    R: RingBuffer<Item = VolatilityBucket>,
{
    data: R,
    aggregation_millis: i64,
    windows: [MaybeUninit<TimedVolatilityWindow>; WINDOWS],
    window_count: usize,
    warmup_policy: WarmupPolicy,
    previous_value: Option<f64>,
    first_timestamp: Option<i64>,
    last_observed_timestamp: Option<i64>,
}

impl<R, const WINDOWS: usize> RollingVolatilityTimed<R, WINDOWS>
where
    R: RingBuffer<Item = VolatilityBucket>,
{
    fn new(data: R, aggregation: Duration, warmup_policy: WarmupPolicy) -> Result<Self> {
        let aggregation_millis = aggregation.as_millis();
        if aggregation_millis == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::AggregationTooShort,
            ));
        }
        if !aggregation.subsec_nanos().is_multiple_of(1_000_000) {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::DurationPrecision {
                    field: DurationField::Aggregation,
                },
            ));
        }
        let aggregation_millis = i64::try_from(aggregation_millis).map_err(|_| {
            FimlError::InvalidArgument(InvalidArgumentError::DurationOutOfRange {
                field: DurationField::Aggregation,
            })
        })?;
        Ok(Self {
            data,
            aggregation_millis,
            windows: [const { MaybeUninit::uninit() }; WINDOWS],
            window_count: 0,
            warmup_policy,
            previous_value: None,
            first_timestamp: None,
            last_observed_timestamp: None,
        })
    }

    /// Adds a window spanning `periods` aggregation buckets.
    ///
    /// # Errors
    ///
    /// Rejects zero periods, periods at least as large as history capacity, durations
    /// outside `i64` milliseconds, additions after data, or more than `WINDOWS` windows.
    pub fn add_window_with_periods(&mut self, periods: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowLimitReached { limit: WINDOWS },
            ));
        }
        if !self.data.is_empty() || self.previous_value.is_some() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowAddedAfterData,
            ));
        }
        if periods == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodZero,
            ));
        }
        if periods >= self.data.capacity() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodMustBeLessThanCapacity {
                    period: periods,
                    capacity: self.data.capacity(),
                },
            ));
        }
        let periods = i64::try_from(periods).map_err(|_| {
            FimlError::InvalidArgument(InvalidArgumentError::WindowPeriodOutOfRange {
                target: IntegerTarget::Signed64,
            })
        })?;
        let duration =
            periods
                .checked_mul(self.aggregation_millis)
                .ok_or(FimlError::InvalidArgument(
                    InvalidArgumentError::WindowDurationOutOfRange,
                ))?;
        self.windows[self.window_count].write(TimedVolatilityWindow {
            duration,
            bucket_count: 0,
            moments: Moments::default(),
            ready: false,
        });
        self.window_count += 1;
        Ok(())
    }

    /// Adds a duration-based window before receiving data.
    ///
    /// # Errors
    ///
    /// The duration must use whole `i64` milliseconds and be a positive multiple of
    /// aggregation. The window-count and capacity limits of [`Self::add_window_with_periods`] apply.
    pub fn add_window_with_duration(&mut self, window: Duration) -> Result<()> {
        if !window.subsec_nanos().is_multiple_of(1_000_000) {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::DurationPrecision {
                    field: DurationField::Window,
                },
            ));
        }
        let window_millis = i64::try_from(window.as_millis()).map_err(|_| {
            FimlError::InvalidArgument(InvalidArgumentError::DurationOutOfRange {
                field: DurationField::Window,
            })
        })?;
        if window_millis < self.aggregation_millis {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowShorterThanAggregation,
            ));
        }
        if window_millis % self.aggregation_millis != 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowNotMultipleOfAggregation,
            ));
        }
        let periods = usize::try_from(window_millis / self.aggregation_millis).map_err(|_| {
            FimlError::InvalidArgument(InvalidArgumentError::WindowPeriodOutOfRange {
                target: IntegerTarget::Usize,
            })
        })?;
        self.add_window_with_periods(periods)
    }

    fn bucket_start(&self, timestamp: i64) -> i64 {
        timestamp - timestamp.rem_euclid(self.aggregation_millis)
    }

    fn update_readiness(&mut self, now: i64) {
        for index in 0..self.window_count {
            let window = unsafe { self.windows[index].assume_init_mut() };
            if !window.ready {
                window.ready = match (self.warmup_policy, self.first_timestamp) {
                    (WarmupPolicy::FirstValue, Some(_)) => true,
                    (WarmupPolicy::FullWindow, Some(first)) => {
                        now.saturating_sub(first) >= window.duration
                    }
                    (_, None) => false,
                };
            }
        }
    }

    /// Advances timed windows without recording a new source value.
    pub(crate) fn observe(&mut self, now: i64) -> bool {
        if self.last_observed_timestamp == Some(now) {
            return false;
        }
        self.last_observed_timestamp = Some(now);
        for index in 0..self.window_count {
            let window = unsafe { self.windows[index].assume_init_mut() };
            while window.bucket_count > 0 {
                let bucket = self
                    .data
                    .peek_front_at(self.data.len() - window.bucket_count)
                    .unwrap();
                if bucket.timestamp + window.duration > now {
                    break;
                }
                window.moments.remove(bucket.moments);
                window.bucket_count -= 1;
            }
        }
        self.update_readiness(now);
        true
    }

    /// Records a value and updates volatility using its simple return from the previous value.
    /// The first value, or a zero previous value, only establishes a new baseline.
    /// The caller must supply finite values. Timestamps are epoch milliseconds and must
    /// be nondecreasing; this method does not validate their ordering.
    pub fn update(&mut self, value: f64, timestamp: i64) {
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(timestamp);
            self.update_readiness(timestamp);
        }
        let _ = self.observe(timestamp);
        let Some(previous_value) = self.previous_value.replace(value) else {
            return;
        };
        if previous_value == 0.0 {
            return;
        }

        let current_return = (value - previous_value) / previous_value;
        let bucket_start = self.bucket_start(timestamp);
        if self
            .data
            .peek_back()
            .is_some_and(|bucket| bucket.timestamp == bucket_start)
        {
            let mut bucket = self.data.pop_back().unwrap();
            bucket.moments.add_value(current_return);
            self.data.push_back(bucket);
            for index in 0..self.window_count {
                let window = unsafe { self.windows[index].assume_init_mut() };
                if window.bucket_count > 0 {
                    window.moments.add_value(current_return);
                } else if bucket_start + window.duration > timestamp {
                    window.moments.add_value(current_return);
                    window.bucket_count = 1;
                }
            }
            return;
        }

        let mut moments = Moments::default();
        moments.add_value(current_return);
        let len_before = self.data.len();
        let evicted = self.data.push_back(VolatilityBucket {
            timestamp: bucket_start,
            moments,
        });
        for index in 0..self.window_count {
            let window = unsafe { self.windows[index].assume_init_mut() };
            if let Some(bucket) = evicted
                && window.bucket_count == len_before
            {
                window.moments.remove(bucket.moments);
                window.bucket_count -= 1;
            }
            window.moments.merge(moments);
            window.bucket_count += 1;
        }
    }

    /// Returns the zero-based window's value, or `None` if absent or still warming up.
    /// Also returns `None` when no return observations are available.
    pub fn value_at(&self, index: usize) -> Option<f64> {
        self.is_ready_at(index)
            .then(|| {
                unsafe { self.windows[index].assume_init_ref() }
                    .moments
                    .volatility()
            })
            .flatten()
    }

    /// Reports whether the zero-based window has met its warm-up policy; false if absent.
    /// Readiness does not guarantee a value when no returns are available.
    pub fn is_ready_at(&self, index: usize) -> bool {
        index < self.window_count && unsafe { self.windows[index].assume_init_ref() }.ready
    }

    /// Returns true when at least one window exists and all windows meet their warm-up policy.
    pub fn is_ready(&self) -> bool {
        self.window_count > 0 && (0..self.window_count).all(|index| self.is_ready_at(index))
    }

    /// Returns window values in insertion order, with `NaN` for unused or unavailable slots.
    pub fn values(&self) -> [f64; WINDOWS] {
        let mut values = [f64::NAN; WINDOWS];
        for (index, value) in values.iter_mut().enumerate().take(self.window_count) {
            if let Some(volatility) = self.value_at(index) {
                *value = volatility;
            }
        }
        values
    }
}

impl<const PERIODS: usize, const WINDOWS: usize>
    RollingVolatilityTimed<StackRingBuffer<PERIODS, VolatilityBucket>, WINDOWS>
{
    /// Creates an empty indicator with inline history; add windows before updating.
    ///
    /// # Errors
    ///
    /// Rejects zero capacity or an aggregation that is not positive whole milliseconds
    /// representable as `i64`. Reserve more history slots than the largest window's bucket count.
    pub fn new_stack(aggregation: Duration, warmup_policy: WarmupPolicy) -> Result<Self> {
        if PERIODS == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::RingBufferCapacityZero,
            ));
        }
        Self::new(new_stack_ring_buffer(), aggregation, warmup_policy)
    }
}

impl<const WINDOWS: usize> RollingVolatilityTimed<HeapRingBuffer<VolatilityBucket>, WINDOWS> {
    /// Creates an empty indicator with heap-allocated history; add windows before updating.
    ///
    /// # Errors
    ///
    /// Rejects zero capacity or an aggregation that is not positive whole milliseconds
    /// representable as `i64`. Reserve more history slots than the largest window's bucket count.
    pub fn new_heap(
        aggregation: Duration,
        capacity: usize,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::RingBufferCapacityZero,
            ));
        }
        Self::new(new_heap_ring_buffer(capacity), aggregation, warmup_policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn sample_windows_calculate_population_volatility_of_returns() {
        let mut volatility: RollingVolatility<StackRingBuffer<3, f64>, 2> =
            RollingVolatility::new_stack(WarmupPolicy::FirstValue);
        volatility.add_window(2).unwrap();
        volatility.add_window(3).unwrap();

        for value in [100.0, 110.0, 99.0, 118.8] {
            volatility.update(value);
        }

        approx_eq(volatility.value_at(0).unwrap(), 0.15);
        approx_eq(volatility.value_at(1).unwrap(), 0.1247219128924647);
    }

    #[test]
    fn sample_and_timed_preserve_small_variance_between_large_returns() {
        let mut volatility: RollingVolatility<StackRingBuffer<2, f64>, 1> =
            RollingVolatility::new_stack(WarmupPolicy::FirstValue);
        volatility.add_window(2).unwrap();

        for value in [1.0, 2.0, 4.00000002] {
            volatility.update(value);
        }

        let value = volatility.value_at(0).unwrap();
        assert!((4.9e-9..5.1e-9).contains(&value), "{value}");

        let mut timed: RollingVolatilityTimed<StackRingBuffer<4, VolatilityBucket>, 1> =
            RollingVolatilityTimed::new_stack(Duration::from_secs(1), WarmupPolicy::FirstValue)
                .unwrap();
        timed.add_window_with_periods(3).unwrap();
        for (timestamp, value) in [1.0, 2.0, 4.00000002].into_iter().enumerate() {
            timed.update(value, timestamp as i64 * 1_000);
        }
        let value = timed.value_at(0).unwrap();
        assert!((4.9e-9..5.1e-9).contains(&value), "{value}");
    }

    #[test]
    fn timed_window_uses_every_return_in_each_bucket_and_expires_old_buckets() {
        let mut volatility: RollingVolatilityTimed<StackRingBuffer<3, VolatilityBucket>, 1> =
            RollingVolatilityTimed::new_stack(Duration::from_secs(1), WarmupPolicy::FirstValue)
                .unwrap();
        volatility.add_window_with_periods(2).unwrap();

        volatility.update(100.0, 0);
        volatility.update(110.0, 100);
        volatility.update(99.0, 200);
        approx_eq(volatility.value_at(0).unwrap(), 0.1);

        volatility.update(118.8, 2_000);
        approx_eq(volatility.value_at(0).unwrap(), 0.0);
    }

    #[test]
    fn zero_previous_value_rebases_without_poisoning_state() {
        let mut volatility: RollingVolatility<StackRingBuffer<2, f64>, 1> =
            RollingVolatility::new_stack(WarmupPolicy::FirstValue);
        volatility.add_window(2).unwrap();

        volatility.update(0.0);
        assert!(volatility.is_ready_at(0));
        assert_eq!(volatility.value_at(0), None);

        volatility.update(50.0);
        assert!(volatility.is_ready_at(0));
        assert_eq!(volatility.value_at(0), None);

        volatility.update(75.0);

        approx_eq(volatility.value_at(0).unwrap(), 0.0);
    }
}

use std::mem::MaybeUninit;
use std::time::Duration;

use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use crate::{DurationField, FimlError, IntegerTarget, InvalidArgumentError, Result, WarmupPolicy};

/// One fixed-duration bucket used by fixed-width rolling timed sums.
pub struct TimedSumBucket<const VALUES: usize> {
    timestamp: i64,
    values: [f64; VALUES],
}

struct TimedSumWindow<const VALUES: usize> {
    duration: i64,
    values: [f64; VALUES],
    ready: bool,
    front_offset: usize,
}

pub(crate) struct RollingTimedSum<R, const WINDOWS: usize, const VALUES: usize>
where
    R: RingBuffer<Item = TimedSumBucket<VALUES>>,
{
    data: R,
    millis_aggregation: i64,
    windows: [MaybeUninit<TimedSumWindow<VALUES>>; WINDOWS],
    window_count: usize,
    warmup_policy: WarmupPolicy,
    first_timestamp: Option<i64>,
    last_observed_timestamp: Option<i64>,
}

impl<const N: usize, const WINDOWS: usize, const VALUES: usize>
    RollingTimedSum<StackRingBuffer<N, TimedSumBucket<VALUES>>, WINDOWS, VALUES>
{
    pub(crate) fn new_stack(aggregation: Duration, warmup_policy: WarmupPolicy) -> Result<Self> {
        if N == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::RingBufferCapacityZero,
            ));
        }
        Self::new_with_buffer(
            new_stack_ring_buffer::<N, TimedSumBucket<VALUES>>(),
            aggregation,
            N,
            warmup_policy,
        )
    }
}

impl<const WINDOWS: usize, const VALUES: usize>
    RollingTimedSum<HeapRingBuffer<TimedSumBucket<VALUES>>, WINDOWS, VALUES>
{
    pub(crate) fn new_heap(
        aggregation: Duration,
        capacity: usize,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::RingBufferCapacityZero,
            ));
        }
        Self::new_with_buffer(
            new_heap_ring_buffer::<TimedSumBucket<VALUES>>(capacity),
            aggregation,
            capacity,
            warmup_policy,
        )
    }
}

impl<R, const WINDOWS: usize, const VALUES: usize> RollingTimedSum<R, WINDOWS, VALUES>
where
    R: RingBuffer<Item = TimedSumBucket<VALUES>>,
{
    fn new_with_buffer(
        data: R,
        aggregation: Duration,
        capacity: usize,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::RingBufferCapacityZero,
            ));
        }
        if aggregation.is_zero() {
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
        let millis_aggregation = i64::try_from(aggregation.as_millis()).map_err(|_| {
            FimlError::InvalidArgument(InvalidArgumentError::DurationOutOfRange {
                field: DurationField::Aggregation,
            })
        })?;
        Ok(Self {
            data,
            millis_aggregation,
            windows: [const { MaybeUninit::<TimedSumWindow<VALUES>>::uninit() }; WINDOWS],
            window_count: 0,
            warmup_policy,
            first_timestamp: None,
            last_observed_timestamp: None,
        })
    }

    pub(crate) fn add_window_with_periods(&mut self, periods: usize) -> Result<()> {
        if self.window_count >= WINDOWS {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowLimitReached { limit: WINDOWS },
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
        if !self.data.is_empty() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowAddedAfterData,
            ));
        }
        let periods = i64::try_from(periods).map_err(|_| {
            FimlError::InvalidArgument(InvalidArgumentError::WindowPeriodOutOfRange {
                target: IntegerTarget::Signed64,
            })
        })?;
        let duration =
            periods
                .checked_mul(self.millis_aggregation)
                .ok_or(FimlError::InvalidArgument(
                    InvalidArgumentError::WindowDurationOutOfRange,
                ))?;
        self.windows[self.window_count].write(TimedSumWindow {
            duration,
            values: [0.0; VALUES],
            ready: false,
            front_offset: 0,
        });
        self.window_count += 1;
        Ok(())
    }

    fn bucket_start(&self, timestamp: i64) -> i64 {
        timestamp - timestamp.rem_euclid(self.millis_aggregation)
    }

    fn expire_old_buckets(&mut self, current_window_start: i64) {
        let data_len = self.data.len();
        for index in 0..self.window_count {
            let window = unsafe { self.windows[index].assume_init_mut() };
            while window.front_offset < data_len {
                let Some(bucket) = self.data.peek_front_at(window.front_offset) else {
                    break;
                };
                if bucket.timestamp + window.duration > current_window_start {
                    break;
                }
                for (sum, value) in window.values.iter_mut().zip(bucket.values) {
                    *sum -= value;
                }
                window.front_offset += 1;
            }
            if window.front_offset == data_len {
                window.values.fill(0.0);
            }
        }
    }

    fn add_to_windows(&mut self, values: [f64; VALUES]) {
        for index in 0..self.window_count {
            let window = unsafe { self.windows[index].assume_init_mut() };
            for (sum, value) in window.values.iter_mut().zip(values) {
                *sum += value;
            }
        }
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

    pub(crate) fn observe(&mut self, now: i64) -> bool {
        if self.last_observed_timestamp == Some(now) {
            return false;
        }
        self.last_observed_timestamp = Some(now);
        self.expire_old_buckets(self.bucket_start(now));
        self.update_readiness(now);
        true
    }

    pub(crate) fn update(&mut self, values: [f64; VALUES], now: i64) {
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(now);
            self.update_readiness(now);
        }
        let _ = self.observe(now);
        let bucket_start = self.bucket_start(now);
        if self
            .data
            .peek_back()
            .is_some_and(|bucket| bucket.timestamp == bucket_start)
        {
            let mut bucket = self.data.pop_back().unwrap();
            for (sum, value) in bucket.values.iter_mut().zip(values) {
                *sum += value;
            }
            self.data.push_back(bucket);
        } else if self
            .data
            .push_back(TimedSumBucket {
                timestamp: bucket_start,
                values,
            })
            .is_some()
        {
            for index in 0..self.window_count {
                let window = unsafe { self.windows[index].assume_init_mut() };
                debug_assert!(window.front_offset > 0);
                window.front_offset = window.front_offset.saturating_sub(1);
            }
        }
        self.add_to_windows(values);
    }

    pub(crate) fn window_value(&self, index: usize, value_index: usize) -> Option<f64> {
        self.is_ready_at(index)
            .then(|| unsafe { self.windows[index].assume_init_ref() })
            .and_then(|window| window.values.get(value_index).copied())
    }

    pub(crate) fn is_ready_at(&self, index: usize) -> bool {
        index < self.window_count && unsafe { self.windows[index].assume_init_ref() }.ready
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.window_count > 0 && (0..self.window_count).all(|index| self.is_ready_at(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_value_readies_after_same_timestamp_observation() {
        let mut sum: RollingTimedSum<HeapRingBuffer<TimedSumBucket<1>>, 1, 1> =
            RollingTimedSum::new_heap(Duration::from_secs(1), 2, WarmupPolicy::FirstValue).unwrap();
        sum.add_window_with_periods(1).unwrap();

        assert!(sum.observe(0));
        sum.update([2.0], 0);

        assert_eq!(sum.window_value(0, 0), Some(2.0));
    }

    #[test]
    fn fully_expired_sums_reset_floating_point_residue() {
        let mut sum: RollingTimedSum<HeapRingBuffer<TimedSumBucket<1>>, 1, 1> =
            RollingTimedSum::new_heap(Duration::from_secs(1), 4, WarmupPolicy::FirstValue).unwrap();
        sum.add_window_with_periods(3).unwrap();
        sum.update([1.0], 0);
        sum.update([0.1], 1_000);
        sum.update([0.2], 2_000);

        sum.observe(5_000);

        assert_eq!(sum.window_value(0, 0), Some(0.0));
    }
}

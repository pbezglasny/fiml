use std::time::Duration;

use crate::indicators::timed_sum::{RollingTimedSum, TimedSumBucket};
use crate::ring_buffer::{HeapRingBuffer, RingBuffer, StackRingBuffer};
use crate::{Result, WarmupPolicy};

/// One fixed-duration bucket of summed trade volume.
pub type TradeVolumeBucket = TimedSumBucket<1>;

/// Rolling trade-volume sums over fixed-duration, time-bucketed windows.
pub struct RollingTradeVolumeTimed<R, const WINDOWS: usize>
where
    R: RingBuffer<Item = TradeVolumeBucket>,
{
    sum: RollingTimedSum<R, WINDOWS, 1>,
}

impl<const N: usize, const WINDOWS: usize>
    RollingTradeVolumeTimed<StackRingBuffer<N, TradeVolumeBucket>, WINDOWS>
{
    /// Creates a stack-backed indicator with room for `N` aggregation buckets.
    pub fn new_stack(aggregation: Duration, warmup_policy: WarmupPolicy) -> Result<Self> {
        Ok(Self {
            sum: RollingTimedSum::new_stack(aggregation, warmup_policy)?,
        })
    }
}

impl<const WINDOWS: usize> RollingTradeVolumeTimed<HeapRingBuffer<TradeVolumeBucket>, WINDOWS> {
    /// Creates a heap-backed indicator with room for `capacity` aggregation buckets.
    pub fn new_heap(
        aggregation: Duration,
        capacity: usize,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self> {
        Ok(Self {
            sum: RollingTimedSum::new_heap(aggregation, capacity, warmup_policy)?,
        })
    }
}

impl<R, const WINDOWS: usize> RollingTradeVolumeTimed<R, WINDOWS>
where
    R: RingBuffer<Item = TradeVolumeBucket>,
{
    /// Adds a rolling window expressed as a number of aggregation buckets.
    pub fn add_window_with_periods(&mut self, periods: usize) -> Result<()> {
        self.sum.add_window_with_periods(periods)
    }

    /// Records trade `volume` at `timestamp` in epoch milliseconds.
    pub(crate) fn update_inner(&mut self, volume: f64, timestamp: i64) {
        self.sum.update([volume], timestamp);
    }

    /// Records trade `volume` at the current system time.
    pub fn update(&mut self, volume: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_millis() as i64;
        self.update_inner(volume, now);
    }

    /// Advances window expiry and warm-up without recording a trade.
    pub(crate) fn observe(&mut self, timestamp: i64) -> bool {
        self.sum.observe(timestamp)
    }

    /// Returns the rolling volume for a configured window when it is ready.
    pub fn window_value(&self, index: usize) -> Option<f64> {
        self.sum.window_value(index, 0)
    }

    /// Returns whether one configured window has completed warm-up.
    pub fn is_ready_at(&self, index: usize) -> bool {
        self.sum.is_ready_at(index)
    }

    /// Returns whether every configured window has completed warm-up.
    pub fn is_ready(&self) -> bool {
        self.sum.is_ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_windows_sum_and_expire_trade_volume() {
        let mut volume: RollingTradeVolumeTimed<HeapRingBuffer<TradeVolumeBucket>, 2> =
            RollingTradeVolumeTimed::new_heap(Duration::from_secs(1), 4, WarmupPolicy::FirstValue)
                .unwrap();
        volume.add_window_with_periods(2).unwrap();
        volume.add_window_with_periods(3).unwrap();

        volume.update_inner(1.5, 0);
        volume.update_inner(2.5, 500);
        volume.update_inner(4.0, 1_000);
        assert_eq!(volume.window_value(0), Some(8.0));
        assert_eq!(volume.window_value(1), Some(8.0));

        volume.update_inner(8.0, 2_000);
        assert_eq!(volume.window_value(0), Some(12.0));
        assert_eq!(volume.window_value(1), Some(16.0));
    }

    #[test]
    fn full_window_warms_on_observed_time_and_empty_window_is_zero() {
        let mut volume: RollingTradeVolumeTimed<HeapRingBuffer<TradeVolumeBucket>, 1> =
            RollingTradeVolumeTimed::new_heap(Duration::from_secs(1), 3, WarmupPolicy::FullWindow)
                .unwrap();
        volume.add_window_with_periods(2).unwrap();
        volume.update_inner(3.0, 0);
        assert_eq!(volume.window_value(0), None);
        volume.observe(2_000);
        assert_eq!(volume.window_value(0), Some(0.0));
    }
}

//! Calculates volume-weighted average price over rolling time windows.
//!
use std::time::Duration;

use crate::indicators::timed_sum::{RollingTimedSum, TimedSumBucket};
use crate::ring_buffer::{HeapRingBuffer, RingBuffer, StackRingBuffer};
use crate::{Result, WarmupPolicy};

/// One fixed-duration bucket used by a rolling VWAP calculation.
pub type VwapBucket = TimedSumBucket<2>;

/// Volume-weighted average trade price over fixed-duration, time-bucketed windows.
pub struct VolumeWeightedAveragePriceTimed<R, const WINDOWS: usize>
where
    R: RingBuffer<Item = VwapBucket>,
{
    sum: RollingTimedSum<R, WINDOWS, 2>,
}

impl<const N: usize, const WINDOWS: usize>
    VolumeWeightedAveragePriceTimed<StackRingBuffer<N, VwapBucket>, WINDOWS>
{
    /// Creates a stack-backed indicator with room for `N` aggregation buckets.
    pub fn new_stack(aggregation: Duration, warmup_policy: WarmupPolicy) -> Result<Self> {
        Ok(Self {
            sum: RollingTimedSum::new_stack(aggregation, warmup_policy)?,
        })
    }
}

impl<const WINDOWS: usize> VolumeWeightedAveragePriceTimed<HeapRingBuffer<VwapBucket>, WINDOWS> {
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

impl<R, const WINDOWS: usize> VolumeWeightedAveragePriceTimed<R, WINDOWS>
where
    R: RingBuffer<Item = VwapBucket>,
{
    /// Adds a rolling window expressed as a number of aggregation buckets.
    pub fn add_window_with_periods(&mut self, periods: usize) -> Result<()> {
        self.sum.add_window_with_periods(periods)
    }

    /// Records a trade at `timestamp` in epoch milliseconds.
    pub(crate) fn update_inner(&mut self, price: f64, volume: f64, timestamp: i64) {
        self.sum.update([price * volume, volume], timestamp);
    }

    /// Records a trade at the current system time.
    pub fn update(&mut self, price: f64, volume: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_millis() as i64;
        self.update_inner(price, volume, now);
    }

    /// Advances window expiry and warm-up without recording a trade.
    pub(crate) fn observe(&mut self, timestamp: i64) -> bool {
        self.sum.observe(timestamp)
    }

    /// Returns VWAP for a configured window when it is ready and non-empty.
    pub fn window_value(&self, index: usize) -> Option<f64> {
        let volume = self.sum.window_value(index, 1)?;
        let notional = self.sum.window_value(index, 0)?;
        (volume != 0.0).then_some(notional / volume)
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
    fn grouped_windows_weight_prices_and_expire_trades() {
        let mut vwap: VolumeWeightedAveragePriceTimed<HeapRingBuffer<VwapBucket>, 2> =
            VolumeWeightedAveragePriceTimed::new_heap(
                Duration::from_secs(1),
                4,
                WarmupPolicy::FirstValue,
            )
            .unwrap();
        vwap.add_window_with_periods(2).unwrap();
        vwap.add_window_with_periods(3).unwrap();

        vwap.update_inner(10.0, 1.0, 0);
        vwap.update_inner(20.0, 3.0, 1_000);
        assert_eq!(vwap.window_value(0), Some(17.5));
        assert_eq!(vwap.window_value(1), Some(17.5));

        vwap.update_inner(30.0, 2.0, 2_000);
        assert_eq!(vwap.window_value(0), Some(24.0));
        assert_eq!(vwap.window_value(1), Some(130.0 / 6.0));
    }

    #[test]
    fn empty_ready_window_has_no_vwap() {
        let mut vwap: VolumeWeightedAveragePriceTimed<HeapRingBuffer<VwapBucket>, 1> =
            VolumeWeightedAveragePriceTimed::new_heap(
                Duration::from_secs(1),
                3,
                WarmupPolicy::FullWindow,
            )
            .unwrap();
        vwap.add_window_with_periods(2).unwrap();
        vwap.update_inner(10.0, 1.0, 0);
        assert_eq!(vwap.window_value(0), None);
        vwap.observe(2_000);
        assert_eq!(vwap.window_value(0), None);
        assert!(vwap.is_ready());
    }

    #[test]
    fn expired_fractional_volumes_do_not_publish_residue() {
        let mut vwap: VolumeWeightedAveragePriceTimed<HeapRingBuffer<VwapBucket>, 1> =
            VolumeWeightedAveragePriceTimed::new_heap(
                Duration::from_secs(1),
                4,
                WarmupPolicy::FirstValue,
            )
            .unwrap();
        vwap.add_window_with_periods(3).unwrap();
        vwap.update_inner(100.0, 1.0, 0);
        vwap.update_inner(101.0, 0.1, 1_000);
        vwap.update_inner(102.0, 0.2, 2_000);

        vwap.observe(5_000);

        assert_eq!(vwap.window_value(0), None);
    }
}

//! Counts trades in a rolling time window.
//!
use std::time::Duration;

use crate::indicators::timed_sum::{RollingTimedSum, TimedSumBucket};
use crate::ring_buffer::{HeapRingBuffer, RingBuffer};
use crate::{
    DurationField, FimlError, IndicatorKind, IntegerTarget, InvalidArgumentError, Result,
    WarmupPolicy,
};

/// One fixed-duration bucket of trade counts.
pub type CountBucket = TimedSumBucket<1>;

/// Number of trades within a single rolling time window.
pub struct TradeCountTimed<R>
where
    R: RingBuffer<Item = CountBucket>,
{
    sum: RollingTimedSum<R, 1, 1>,
}

impl TradeCountTimed<HeapRingBuffer<CountBucket>> {
    /// Build a heap-backed timed trade counter over `window`, bucketed by
    /// `aggregation`.
    pub fn new_heap(
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self> {
        let periods = validate_durations(aggregation, window)?;
        let capacity = periods.checked_add(1).ok_or(FimlError::InvalidArgument(
            InvalidArgumentError::TimedPeriodTooLarge {
                indicator: IndicatorKind::TradeCountTimed,
            },
        ))?;
        let mut sum = RollingTimedSum::new_heap(aggregation, capacity, warmup_policy)?;
        sum.add_window_with_periods(periods)?;
        Ok(Self { sum })
    }
}

impl<R> TradeCountTimed<R>
where
    R: RingBuffer<Item = CountBucket>,
{
    /// Record one trade at `now` (epoch milliseconds).
    pub(crate) fn update(&mut self, now: i64) {
        self.sum.update([1.0], now);
    }

    /// Advance the indicator to `now` without recording a trade.
    pub(crate) fn observe(&mut self, now: i64) -> bool {
        self.sum.observe(now)
    }

    /// Current rolling trade count over the window.
    pub fn window_value(&self) -> Option<f64> {
        self.sum.window_value(0, 0)
    }

    pub fn is_ready_at(&self, index: usize) -> bool {
        self.sum.is_ready_at(index)
    }

    pub fn is_ready(&self) -> bool {
        self.sum.is_ready()
    }
}

fn validate_durations(aggregation: Duration, window: Duration) -> Result<usize> {
    let aggregation_millis = aggregation.as_millis();
    let window_millis = window.as_millis();
    if !aggregation.subsec_nanos().is_multiple_of(1_000_000)
        || !window.subsec_nanos().is_multiple_of(1_000_000)
    {
        return Err(FimlError::InvalidArgument(
            InvalidArgumentError::DurationPrecision {
                field: DurationField::TimedWindow,
            },
        ));
    }
    if aggregation_millis == 0 {
        return Err(FimlError::InvalidArgument(
            InvalidArgumentError::AggregationTooShort,
        ));
    }
    if window_millis < aggregation_millis {
        return Err(FimlError::InvalidArgument(
            InvalidArgumentError::WindowShorterThanAggregation,
        ));
    }
    if !window_millis.is_multiple_of(aggregation_millis) {
        return Err(FimlError::InvalidArgument(
            InvalidArgumentError::WindowNotMultipleOfAggregation,
        ));
    }
    i64::try_from(aggregation_millis).map_err(|_| {
        FimlError::InvalidArgument(InvalidArgumentError::DurationOutOfRange {
            field: DurationField::Aggregation,
        })
    })?;
    i64::try_from(window_millis).map_err(|_| {
        FimlError::InvalidArgument(InvalidArgumentError::DurationOutOfRange {
            field: DurationField::Window,
        })
    })?;
    usize::try_from(window_millis / aggregation_millis).map_err(|_| {
        FimlError::InvalidArgument(InvalidArgumentError::WindowPeriodOutOfRange {
            target: IntegerTarget::Usize,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter(
        window: u64,
        warmup_policy: WarmupPolicy,
    ) -> TradeCountTimed<HeapRingBuffer<CountBucket>> {
        TradeCountTimed::new_heap(
            Duration::from_secs(1),
            Duration::from_secs(window),
            warmup_policy,
        )
        .unwrap()
    }

    #[test]
    fn counts_and_expires_trade_buckets() {
        let mut counter = counter(2, WarmupPolicy::FirstValue);
        counter.update(0);
        counter.update(100);
        counter.update(1_000);
        assert_eq!(counter.window_value(), Some(3.0));
        counter.update(2_000);
        assert_eq!(counter.window_value(), Some(2.0));
    }

    #[test]
    fn full_window_warms_and_empty_window_is_zero() {
        let mut counter = counter(2, WarmupPolicy::FullWindow);
        counter.update(0);
        assert_eq!(counter.window_value(), None);
        counter.observe(2_000);
        assert_eq!(counter.window_value(), Some(0.0));
    }

    #[test]
    fn survives_ring_eviction() {
        let mut counter = counter(2, WarmupPolicy::FirstValue);
        for timestamp in (0..10).map(|value| value * 1_000) {
            counter.update(timestamp);
        }
        assert_eq!(counter.window_value(), Some(2.0));
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(
            TradeCountTimed::<HeapRingBuffer<CountBucket>>::new_heap(
                Duration::ZERO,
                Duration::from_secs(1),
                WarmupPolicy::FirstValue,
            )
            .is_err()
        );
        assert!(
            TradeCountTimed::<HeapRingBuffer<CountBucket>>::new_heap(
                Duration::from_secs(1),
                Duration::from_millis(1_500),
                WarmupPolicy::FirstValue,
            )
            .is_err()
        );
    }
}

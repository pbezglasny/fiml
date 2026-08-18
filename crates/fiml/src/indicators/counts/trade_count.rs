use std::marker::PhantomData;
use std::time::Duration;

use crate::ring_buffer::{HeapRingBuffer, RingBuffer, new_heap_ring_buffer};
use crate::{FimlError, Float, Result, WarmupPolicy};

/// One fixed-duration bucket: the trades that fell into `[timestamp, timestamp +
/// aggregation)`, counted.
pub struct CountBucket {
    timestamp: i64,
    count: u64,
}

/// Number of trades within a single rolling time window.
///
/// Trades are aggregated into fixed-duration buckets and the window exposes the
/// rolling sum of the bucket counts. This mirrors the bucketing of
/// [`OnBalanceVolumeTimed`](crate::indicators::OnBalanceVolumeTimed) but sums a
/// plain per-bucket trade count instead of signed volume, so it carries a single
/// window rather than a configurable set.
pub struct TradeCountTimed<R, F>
where
    R: RingBuffer<Item = CountBucket>,
    F: Float,
{
    data: R,
    millis_aggregation: i64,
    window_duration: i64,
    /// Running sum of bucket counts inside the window.
    window_count: u64,
    /// Front-relative index of the oldest bucket still inside the window. Buckets
    /// before it have expired and were already subtracted from `window_count`.
    front_offset: usize,
    warmup_policy: WarmupPolicy,
    first_timestamp: Option<i64>,
    last_observed_timestamp: Option<i64>,
    ready: bool,
    _marker: PhantomData<F>,
}

impl<F> TradeCountTimed<HeapRingBuffer<CountBucket>, F>
where
    F: Float,
{
    /// Build a heap-backed timed trade counter over `window`, bucketed by
    /// `aggregation`. Both durations are in milliseconds; `window` must be a
    /// non-zero multiple of a non-zero `aggregation`.
    pub fn new_heap(
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self> {
        let periods = validate_durations(aggregation, window)?;
        // One extra slot so the oldest bucket has expired from the window before
        // the ring evicts it (mirrors the OBV invariant).
        let capacity = periods
            .checked_add(1)
            .ok_or_else(|| FimlError::InvalidArgument("trade count window is too large".into()))?;
        let data = new_heap_ring_buffer::<CountBucket>(capacity);
        Ok(Self {
            data,
            millis_aggregation: i64::try_from(aggregation.as_millis()).map_err(|_| {
                FimlError::InvalidArgument(
                    "trade count aggregation must fit signed 64-bit milliseconds".into(),
                )
            })?,
            window_duration: i64::try_from(window.as_millis()).map_err(|_| {
                FimlError::InvalidArgument(
                    "trade count window must fit signed 64-bit milliseconds".into(),
                )
            })?,
            window_count: 0,
            front_offset: 0,
            warmup_policy,
            first_timestamp: None,
            last_observed_timestamp: None,
            ready: false,
            _marker: PhantomData,
        })
    }
}

impl<R, F> TradeCountTimed<R, F>
where
    R: RingBuffer<Item = CountBucket>,
    F: Float,
{
    fn bucket_start(&self, timestamp: i64) -> i64 {
        timestamp - timestamp.rem_euclid(self.millis_aggregation)
    }

    fn expire_old_buckets(&mut self, current_window_start: i64) {
        while self.front_offset < self.data.len() {
            let Some(bucket) = self.data.peek_front_at(self.front_offset) else {
                break;
            };
            if bucket.timestamp + self.window_duration > current_window_start {
                break;
            }
            self.window_count -= bucket.count;
            self.front_offset += 1;
        }
    }

    /// Record one trade at `now` (epoch milliseconds).
    pub(crate) fn update(&mut self, event_timestamp: i64) {
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(event_timestamp);
            self.update_readiness(event_timestamp);
        }
        let _ = self.observe(event_timestamp);
        let insert_bucket_start = self.bucket_start(event_timestamp);

        // Same bucket as the last trade? Increment its count in place.
        if self
            .data
            .peek_back()
            .is_some_and(|bucket| bucket.timestamp == insert_bucket_start)
        {
            let mut bucket = self.data.pop_back().unwrap();
            bucket.count += 1;
            self.data.push_back(bucket);
            self.window_count += 1;
        } else {
            let bucket = CountBucket {
                timestamp: insert_bucket_start,
                count: 1,
            };
            if self.data.push_back(bucket).is_some() {
                // The evicted front bucket already expired from the window, so the
                // cursor only needs shifting back by one.
                self.front_offset = self.front_offset.saturating_sub(1);
            }
            self.window_count += 1;
        }
    }

    fn update_readiness(&mut self, now: i64) {
        if !self.ready {
            self.ready = match (self.warmup_policy, self.first_timestamp) {
                (WarmupPolicy::FirstValue, Some(_)) => true,
                (WarmupPolicy::FullWindow, Some(first)) => {
                    now.saturating_sub(first) >= self.window_duration
                }
                (_, None) => false,
            };
        }
    }

    /// Advance the indicator to `now` without recording a new trade.
    ///
    /// This expires old buckets and may complete full-window warm-up when an
    /// unrelated event advances global event time. Returns `true` when the
    /// timestamp was newly observed, or `false` when it was already processed.
    pub(crate) fn observe(&mut self, now: i64) -> bool {
        if self.last_observed_timestamp == Some(now) {
            return false;
        }
        self.last_observed_timestamp = Some(now);
        self.expire_old_buckets(self.bucket_start(now));
        self.update_readiness(now);
        true
    }

    /// Current rolling trade count over the window.
    pub fn window_value(&self) -> Option<F> {
        self.ready
            .then(|| F::from_usize(self.window_count as usize))
    }

    pub fn is_ready_at(&self, index: usize) -> bool {
        index == 0 && self.ready
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

pub(crate) fn validate_durations(aggregation: Duration, window: Duration) -> Result<usize> {
    let aggregation_millis = aggregation.as_millis();
    let window_millis = window.as_millis();
    if !aggregation.subsec_nanos().is_multiple_of(1_000_000)
        || !window.subsec_nanos().is_multiple_of(1_000_000)
    {
        return Err(FimlError::InvalidArgument(
            "trade count durations must use whole-millisecond precision".to_string(),
        ));
    }
    if aggregation_millis == 0 {
        return Err(FimlError::InvalidArgument(
            "trade count aggregation must be at least 1 millisecond".to_string(),
        ));
    }
    if window_millis < aggregation_millis {
        return Err(FimlError::InvalidArgument(
            "trade count window cannot be less than aggregation".to_string(),
        ));
    }
    if !window_millis.is_multiple_of(aggregation_millis) {
        return Err(FimlError::InvalidArgument(
            "trade count window must be a multiple of aggregation".to_string(),
        ));
    }
    i64::try_from(aggregation_millis).map_err(|_| {
        FimlError::InvalidArgument(
            "trade count aggregation must fit signed 64-bit milliseconds".to_string(),
        )
    })?;
    i64::try_from(window_millis).map_err(|_| {
        FimlError::InvalidArgument(
            "trade count window must fit signed 64-bit milliseconds".to_string(),
        )
    })?;
    usize::try_from(window_millis / aggregation_millis)
        .map_err(|_| FimlError::InvalidArgument("trade count period must fit usize".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn counts_trades_in_the_same_bucket() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(2_000),
                WarmupPolicy::FirstValue,
            )
            .unwrap();

        counter.update(0);
        counter.update(100);
        counter.update(900);

        assert!(approx_eq(counter.window_value().unwrap(), 3.0));
    }

    #[test]
    fn full_window_readiness_advances_with_time_and_empty_window_is_zero() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(2_000),
                WarmupPolicy::FullWindow,
            )
            .unwrap();

        counter.update(0);
        counter.update(1_000);
        assert!(counter.observe(1_999));
        assert!(!counter.is_ready());
        assert_eq!(counter.window_value(), None);

        assert!(counter.observe(2_000));
        assert!(counter.is_ready());
        assert_eq!(counter.window_value(), Some(1.0));

        assert!(counter.observe(3_000));
        assert!(counter.is_ready());
        assert_eq!(counter.window_value(), Some(0.0));
    }

    #[test]
    fn sums_counts_across_buckets_in_window() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(3_000),
                WarmupPolicy::FirstValue,
            )
            .unwrap();

        counter.update(0); // bucket 0
        counter.update(1_000); // bucket 1
        counter.update(1_500); // bucket 1
        counter.update(2_000); // bucket 2

        assert!(approx_eq(counter.window_value().unwrap(), 4.0));
    }

    #[test]
    fn old_buckets_expire_from_window() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(2_000),
                WarmupPolicy::FirstValue,
            )
            .unwrap();

        counter.update(0); // bucket 0
        counter.update(1_000); // bucket 1
        counter.update(2_000); // bucket 2 -> bucket 0 now outside 2s window
        counter.update(3_000); // bucket 3 -> bucket 1 now outside window

        // Window keeps the last two buckets (2 and 3): one trade each.
        assert!(approx_eq(counter.window_value().unwrap(), 2.0));
    }

    #[test]
    fn survives_ring_eviction() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(2_000),
                WarmupPolicy::FirstValue,
            )
            .unwrap();

        for i in 0..10 {
            counter.update(i * 1_000);
        }

        // Only the last two buckets remain inside the 2s window.
        assert!(approx_eq(counter.window_value().unwrap(), 2.0));
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(
            TradeCountTimed::<HeapRingBuffer<CountBucket>, f64>::new_heap(
                Duration::ZERO,
                Duration::from_millis(1_000),
                WarmupPolicy::FirstValue,
            )
            .is_err()
        );
        assert!(
            TradeCountTimed::<HeapRingBuffer<CountBucket>, f64>::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(1_500),
                WarmupPolicy::FirstValue,
            )
            .is_err()
        );
    }
}

use std::marker::PhantomData;
use std::time::Duration;

use crate::ring_buffer::{HeapRingBuffer, RingBuffer, new_heap_ring_buffer};
use crate::{FimlError, Float, Result};

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
    _marker: PhantomData<F>,
}

impl<F> TradeCountTimed<HeapRingBuffer<CountBucket>, F>
where
    F: Float,
{
    /// Build a heap-backed timed trade counter over `window`, bucketed by
    /// `aggregation`. Both durations are in milliseconds; `window` must be a
    /// non-zero multiple of a non-zero `aggregation`.
    pub fn new_heap(aggregation: Duration, window: Duration) -> Result<Self> {
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
    pub(crate) fn update_inner(&mut self, now: i64) {
        let insert_bucket_start = self.bucket_start(now);

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
            self.expire_old_buckets(insert_bucket_start);
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

    /// Record one trade at the current wall-clock time.
    pub fn update(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as i64;
        self.update_inner(now);
    }

    /// Current rolling trade count over the window.
    pub fn window_value(&self) -> F {
        F::from_usize(self.window_count as usize)
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
            TradeCountTimed::new_heap(Duration::from_millis(1_000), Duration::from_millis(2_000))
                .unwrap();

        counter.update_inner(0);
        counter.update_inner(100);
        counter.update_inner(900);

        assert!(approx_eq(counter.window_value(), 3.0));
    }

    #[test]
    fn sums_counts_across_buckets_in_window() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(Duration::from_millis(1_000), Duration::from_millis(3_000))
                .unwrap();

        counter.update_inner(0); // bucket 0
        counter.update_inner(1_000); // bucket 1
        counter.update_inner(1_500); // bucket 1
        counter.update_inner(2_000); // bucket 2

        assert!(approx_eq(counter.window_value(), 4.0));
    }

    #[test]
    fn old_buckets_expire_from_window() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(Duration::from_millis(1_000), Duration::from_millis(2_000))
                .unwrap();

        counter.update_inner(0); // bucket 0
        counter.update_inner(1_000); // bucket 1
        counter.update_inner(2_000); // bucket 2 -> bucket 0 now outside 2s window
        counter.update_inner(3_000); // bucket 3 -> bucket 1 now outside window

        // Window keeps the last two buckets (2 and 3): one trade each.
        assert!(approx_eq(counter.window_value(), 2.0));
    }

    #[test]
    fn survives_ring_eviction() {
        let mut counter: TradeCountTimed<HeapRingBuffer<CountBucket>, f64> =
            TradeCountTimed::new_heap(Duration::from_millis(1_000), Duration::from_millis(2_000))
                .unwrap();

        for i in 0..10 {
            counter.update_inner(i * 1_000);
        }

        // Only the last two buckets remain inside the 2s window.
        assert!(approx_eq(counter.window_value(), 2.0));
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(
            TradeCountTimed::<HeapRingBuffer<CountBucket>, f64>::new_heap(
                Duration::ZERO,
                Duration::from_millis(1_000)
            )
            .is_err()
        );
        assert!(
            TradeCountTimed::<HeapRingBuffer<CountBucket>, f64>::new_heap(
                Duration::from_millis(1_000),
                Duration::from_millis(1_500)
            )
            .is_err()
        );
    }
}

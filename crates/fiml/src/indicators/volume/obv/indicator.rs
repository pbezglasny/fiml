use std::mem::MaybeUninit;
use std::time::Duration;

use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use crate::{FimlError, Float, Result};

struct ObvWindowTimed<F: Float> {
    duration: i64,
    value: F,
    /// Front-relative index of the oldest bucket still inside this window.
    /// Buckets before it are already expired and subtracted from `value`; since
    /// the ring only grows at the back, expiry resumes from this cursor instead
    /// of rescanning the buffer from the front on every trade.
    front_offset: usize,
}

pub struct ObvBucket<F: Float> {
    timestamp: i64,
    close_price: F,
    commulative_volume: F,
    sign: F,
}

impl<F: Float> ObvBucket<F> {
    #[inline]
    fn signed_volume(&self) -> F {
        self.commulative_volume.mul(self.sign)
    }
}

/// On-balance volume (OBV) with time-bucketed rolling windows.
///
/// Trades are aggregated into fixed-duration buckets. Each bucket's total
/// volume is signed by comparing its close price with the previous bucket's
/// close price, and each configured window exposes the rolling sum
/// of those signed bucket deltas.
pub struct OnBalanceVolumeTimed<R, F, const WINDOWS: usize>
where
    R: RingBuffer<Item = ObvBucket<F>>,
    F: Float,
{
    data: R,
    millis_aggregation: i64,
    windows: [MaybeUninit<ObvWindowTimed<F>>; WINDOWS],
    window_count: usize,
}

impl<const N: usize, F, const WINDOWS: usize>
    OnBalanceVolumeTimed<StackRingBuffer<N, ObvBucket<F>>, F, WINDOWS>
where
    F: Float,
{
    pub fn new_stack(aggregation: Duration) -> Result<Self> {
        if N == 0 {
            return Err(FimlError::InvalidArgument(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }
        let stack_data = new_stack_ring_buffer::<N, ObvBucket<F>>();
        Self::new_with_buffer(stack_data, aggregation, N)
    }
}

impl<F, const WINDOWS: usize> OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, WINDOWS>
where
    F: Float,
{
    pub fn new_heap(aggregation: Duration, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }
        let heap_data = new_heap_ring_buffer::<ObvBucket<F>>(capacity);
        Self::new_with_buffer(heap_data, aggregation, capacity)
    }
}

impl<R, F, const WINDOWS: usize> OnBalanceVolumeTimed<R, F, WINDOWS>
where
    R: RingBuffer<Item = ObvBucket<F>>,
    F: Float,
{
    fn new_with_buffer(data: R, aggregation: Duration, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(FimlError::InvalidArgument(
                "Ring buffer capacity must be greater than 0".to_string(),
            ));
        }
        let aggregation_millis = aggregation.as_millis();
        if aggregation_millis == 0 {
            return Err(FimlError::InvalidArgument(
                "Aggregation duration must be at least 1 millisecond".to_string(),
            ));
        }
        if !aggregation.subsec_nanos().is_multiple_of(1_000_000) {
            return Err(FimlError::InvalidArgument(
                "Aggregation duration must use whole-millisecond precision".to_string(),
            ));
        }
        let millis_aggregation = i64::try_from(aggregation_millis).map_err(|_| {
            FimlError::InvalidArgument(
                "Aggregation duration must fit signed 64-bit milliseconds".to_string(),
            )
        })?;
        Ok(Self {
            data,
            millis_aggregation,
            windows: [const { MaybeUninit::<ObvWindowTimed<F>>::uninit() }; WINDOWS],
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
        if self.data.len() > 0 {
            return Err(FimlError::InvalidArgument(
                "Cannot add window after data has been added".to_string(),
            ));
        }

        let periods_i64 = i64::try_from(periods).map_err(|_| {
            FimlError::InvalidArgument("Window period must fit signed 64-bit".to_string())
        })?;
        let duration = periods_i64
            .checked_mul(self.millis_aggregation)
            .ok_or_else(|| {
                FimlError::InvalidArgument(
                    "Window duration must fit signed 64-bit milliseconds".to_string(),
                )
            })?;
        self.windows[self.window_count].write(ObvWindowTimed {
            duration,
            value: F::ZERO,
            front_offset: 0,
        });
        self.window_count += 1;

        #[cfg(feature = "tracing")]
        tracing::debug!(
            indicator = "OBV timed",
            window_index = self.window_count - 1,
            window_count = self.window_count,
            window_capacity = WINDOWS,
            periods,
            duration_millis = duration,
            "added indicator window"
        );
        Ok(())
    }

    fn bucket_start(&self, timestamp: i64) -> i64 {
        timestamp - timestamp.rem_euclid(self.millis_aggregation)
    }

    #[inline]
    fn sign(prev_price: Option<F>, current_price: F) -> F {
        if let Some(prev_price) = prev_price {
            if current_price > prev_price {
                F::ONE
            } else if current_price < prev_price {
                F::ZERO.sub(F::ONE)
            } else {
                F::ZERO
            }
        } else {
            F::ZERO
        }
    }

    fn expire_old_buckets(&mut self, current_window_start: i64) {
        for window_index in 0..self.window_count {
            let window = unsafe { self.windows[window_index].assume_init_mut() };

            // Buckets are time-ordered and the cursor only advances, so resume
            // from the oldest bucket still inside the window rather than from
            // the front of the ring.
            while window.front_offset < self.data.len() {
                let Some(bucket) = self.data.peek_front_at(window.front_offset) else {
                    break;
                };
                if bucket.timestamp + window.duration > current_window_start {
                    break;
                }
                window.value = window.value.sub(bucket.signed_volume());
                window.front_offset += 1;
            }
        }
    }

    /// Shift every window cursor back by one after the ring evicts its front
    /// bucket. That bucket is the oldest, so `periods < capacity` guarantees it
    /// already expired from every window and was subtracted from each value.
    fn shift_front_offsets_after_eviction(&mut self) {
        for window_index in 0..self.window_count {
            let window = unsafe { self.windows[window_index].assume_init_mut() };
            debug_assert!(
                window.front_offset > 0,
                "evicted a bucket that is still inside a window",
            );
            window.front_offset = window.front_offset.saturating_sub(1);
        }
    }

    fn add_delta_to_windows(&mut self, delta: F) {
        for window_index in 0..self.window_count {
            let window = unsafe { self.windows[window_index].assume_init_mut() };
            window.value = window.value.add(delta);
        }
    }

    pub(crate) fn update_inner(&mut self, price: F, volume: F, now: i64) {
        let insert_bucket_start = self.bucket_start(now);

        // are we in the same bucket as the last trade? if so, update the last bucket instead of
        // creating a new one
        if self
            .data
            .peek_back()
            .is_some_and(|bucket| bucket.timestamp == insert_bucket_start)
        {
            let previous_close = self.data.peek_back_at(1).map(|bucket| bucket.close_price);
            let sign = Self::sign(previous_close, price);
            let mut bucket = self.data.pop_back().unwrap();
            let new_volume = bucket.commulative_volume.add(volume);
            let delta = sign.mul(new_volume).sub(bucket.signed_volume());
            bucket.sign = sign;
            bucket.commulative_volume = new_volume;
            bucket.close_price = price;
            self.data.push_back(bucket);
            self.add_delta_to_windows(delta);
        } else {
            self.expire_old_buckets(insert_bucket_start);
            let previous_close = self.data.peek_back().map(|bucket| bucket.close_price);
            let sign = Self::sign(previous_close, price);
            let bucket = ObvBucket {
                timestamp: insert_bucket_start,
                close_price: price,
                commulative_volume: volume,
                sign,
            };
            if self.data.push_back(bucket).is_some() {
                self.shift_front_offsets_after_eviction();
            }
            self.add_delta_to_windows(volume.mul(sign));
        }
    }

    pub fn update(&mut self, price: F, volume: F) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as i64;
        self.update_inner(price, volume, now);
    }

    pub fn window_value(&self, window_idx: usize) -> Option<F> {
        if window_idx >= self.window_count {
            return None;
        }
        let window = unsafe { self.windows[window_idx].assume_init_ref() };
        Some(window.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn first_trade_initializes_price_without_volume_delta() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();

        obv.update_inner(100.0, 10.0, 0);

        assert!(approx_eq(obv.window_value(0).unwrap(), 0.0));
    }

    #[test]
    fn rising_falling_and_equal_prices_create_signed_volume() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 5).unwrap();
        obv.add_window_with_periods(4).unwrap();

        obv.update_inner(100.0, 10.0, 0);
        obv.update_inner(101.0, 7.0, 1_000);
        obv.update_inner(101.0, 99.0, 2_000);
        obv.update_inner(99.0, 3.0, 3_000);

        assert!(approx_eq(obv.window_value(0).unwrap(), 4.0));
    }

    #[test]
    fn first_bucket_keeps_zero_delta_for_same_bucket_trades() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();

        obv.update_inner(100.0, 10.0, 0);
        obv.update_inner(101.0, 7.0, 100);
        obv.update_inner(102.0, 5.0, 900);

        assert!(approx_eq(obv.window_value(0).unwrap(), 0.0));
    }

    #[test]
    fn same_bucket_update_recomputes_aggregate_delta() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();

        obv.update_inner(100.0, 10.0, 0);
        obv.update_inner(102.0, 2.0, 900);
        obv.update_inner(100.0, 5.0, 1_000);
        assert!(approx_eq(obv.window_value(0).unwrap(), -5.0));

        obv.update_inner(104.0, 3.0, 1_500);
        assert!(approx_eq(obv.window_value(0).unwrap(), 8.0));
    }

    #[test]
    fn bucket_delta_compares_close_prices_not_average_prices() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();

        obv.update_inner(100.0, 1.0, 0);
        obv.update_inner(90.0, 1.0, 900);
        obv.update_inner(92.0, 5.0, 1_000);

        assert!(approx_eq(obv.window_value(0).unwrap(), 5.0));
    }

    #[test]
    fn old_buckets_expire_from_window_sum() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();

        obv.update_inner(100.0, 10.0, 0);
        obv.update_inner(101.0, 7.0, 1_000);
        obv.update_inner(102.0, 5.0, 2_000);
        obv.update_inner(103.0, 3.0, 3_000);

        assert!(approx_eq(obv.window_value(0).unwrap(), 8.0));
    }

    #[test]
    fn multiple_windows_share_one_indicator() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 2> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        obv.add_window_with_periods(1).unwrap();
        obv.add_window_with_periods(3).unwrap();

        obv.update_inner(100.0, 10.0, 0);
        obv.update_inner(101.0, 7.0, 1_000);
        obv.update_inner(102.0, 5.0, 2_000);
        obv.update_inner(99.0, 2.0, 3_000);

        assert!(approx_eq(obv.window_value(0).unwrap(), -2.0));
        assert!(approx_eq(obv.window_value(1).unwrap(), 10.0));
    }

    #[test]
    fn cursor_survives_ring_eviction_across_windows() {
        // Capacity is well above both periods, so buckets stay in the ring for
        // several trades after expiring and the front is evicted repeatedly.
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 2> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 5).unwrap();
        obv.add_window_with_periods(2).unwrap();
        obv.add_window_with_periods(4).unwrap();

        // Strictly rising price => every bucket after the first signs its volume
        // positively. Distinct volumes make double-counting observable.
        for (i, volume) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_iter().enumerate() {
            obv.update_inner(100.0 + i as f64, volume, i as i64 * 1_000);
        }

        // 2-period window keeps the last two buckets: 6 + 7.
        assert!(approx_eq(obv.window_value(0).unwrap(), 13.0));
        // 4-period window keeps the last four buckets: 4 + 5 + 6 + 7.
        assert!(approx_eq(obv.window_value(1).unwrap(), 22.0));
    }

    #[test]
    fn rejects_window_after_data_has_been_added() {
        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 2> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 4).unwrap();
        obv.add_window_with_periods(2).unwrap();

        obv.update_inner(100.0, 10.0, 0);
        obv.update_inner(101.0, 7.0, 1_000);

        assert!(obv.add_window_with_periods(3).is_err());
    }

    #[test]
    fn rejects_invalid_configuration() {
        let zero_aggregation: Result<OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1>> =
            OnBalanceVolumeTimed::new_heap(Duration::ZERO, 2);
        assert!(zero_aggregation.is_err());

        let zero_capacity: Result<OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1>> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1), 0);
        assert!(zero_capacity.is_err());

        let mut obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<f64>>, f64, 1> =
            OnBalanceVolumeTimed::new_heap(Duration::from_millis(1), 2).unwrap();
        assert!(obv.add_window_with_periods(0).is_err());
        assert!(obv.add_window_with_periods(2).is_err());
        assert!(obv.add_window_with_periods(1).is_ok());
        assert!(obv.add_window_with_periods(1).is_err());
    }
}

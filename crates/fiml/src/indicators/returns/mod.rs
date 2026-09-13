//! Simple and logarithmic returns over sample lags.

use crate::ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
use crate::{FimlError, InvalidArgumentError, Result};

/// Formula used to compare the current value with a lagged sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReturnKind {
    /// Percentage change: `(current - previous) / previous`.
    Simple,
    /// Continuously compounded change: `ln(current) - ln(previous)`.
    Log,
}

/// Streaming simple or logarithmic returns over several sample lags.
///
/// One ring buffer is shared by every configured lag. An output becomes ready
/// after `lag + 1` samples. Simple returns are unavailable for a zero lagged
/// value; log returns are unavailable unless both values are positive.
pub struct SampleReturns<R, const LAGS: usize>
where
    R: RingBuffer<Item = f64>,
{
    data: R,
    lags: [usize; LAGS],
    lag_count: usize,
    kind: ReturnKind,
}

impl<R, const LAGS: usize> SampleReturns<R, LAGS>
where
    R: RingBuffer<Item = f64>,
{
    fn new(data: R, kind: ReturnKind) -> Self {
        Self {
            data,
            lags: [0; LAGS],
            lag_count: 0,
            kind,
        }
    }

    /// Adds a positive lag before the first sample is observed.
    pub fn add_lag(&mut self, lag: usize) -> Result<()> {
        if self.lag_count >= LAGS {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowLimitReached { limit: LAGS },
            ));
        }
        if !self.data.is_empty() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowAddedAfterData,
            ));
        }
        if lag == 0 {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodZero,
            ));
        }
        if lag >= self.data.capacity() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::WindowPeriodMustBeLessThanCapacity {
                    period: lag,
                    capacity: self.data.capacity(),
                },
            ));
        }
        self.lags[self.lag_count] = lag;
        self.lag_count += 1;
        Ok(())
    }

    /// Records the next sample.
    pub fn update(&mut self, value: f64) {
        self.data.push_back(value);
    }

    /// Returns the latest value for one configured lag, if ready and defined.
    pub fn value_at(&self, index: usize) -> Option<f64> {
        if index >= self.lag_count {
            return None;
        }
        let lag = self.lags[index];
        let &current = self.data.peek_back()?;
        let &previous = self.data.peek_back_at(lag)?;
        match self.kind {
            ReturnKind::Simple if previous != 0.0 => Some((current - previous) / previous),
            ReturnKind::Log if current > 0.0 && previous > 0.0 => {
                Some(current.ln() - previous.ln())
            }
            _ => None,
        }
    }

    /// Reports whether enough samples exist for one configured lag.
    pub fn is_ready_at(&self, index: usize) -> bool {
        index < self.lag_count && self.data.len() > self.lags[index]
    }

    /// Reports whether every configured lag has enough samples.
    pub fn is_ready(&self) -> bool {
        self.lag_count > 0 && (0..self.lag_count).all(|index| self.is_ready_at(index))
    }

    /// Returns all configured outputs, using `NaN` for unavailable values.
    pub fn values(&self) -> [f64; LAGS] {
        let mut values = [f64::NAN; LAGS];
        for (index, value) in values.iter_mut().enumerate().take(self.lag_count) {
            if let Some(result) = self.value_at(index) {
                *value = result;
            }
        }
        values
    }
}

impl<const SAMPLES: usize, const LAGS: usize> SampleReturns<StackRingBuffer<SAMPLES, f64>, LAGS> {
    /// Creates stack-backed returns retaining `SAMPLES` values.
    pub fn new_stack(kind: ReturnKind) -> Self {
        Self::new(new_stack_ring_buffer(), kind)
    }
}

impl<const LAGS: usize> SampleReturns<HeapRingBuffer<f64>, LAGS> {
    /// Creates heap-backed returns retaining `samples` values.
    pub fn new_heap(samples: usize, kind: ReturnKind) -> Self {
        Self::new(new_heap_ring_buffer(samples), kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_simple_and_log_returns_over_multiple_lags() {
        let mut simple: SampleReturns<StackRingBuffer<4, f64>, 2> =
            SampleReturns::new_stack(ReturnKind::Simple);
        simple.add_lag(1).unwrap();
        simple.add_lag(3).unwrap();
        let mut log: SampleReturns<StackRingBuffer<4, f64>, 1> =
            SampleReturns::new_stack(ReturnKind::Log);
        log.add_lag(2).unwrap();

        for value in [100.0, 110.0, 121.0, 133.1] {
            simple.update(value);
            log.update(value);
        }

        assert!((simple.value_at(0).unwrap() - 0.1).abs() < 1e-12);
        assert!((simple.value_at(1).unwrap() - 0.331).abs() < 1e-12);
        assert!((log.value_at(0).unwrap() - 1.21_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn warmup_and_invalid_domains_produce_missing_values() {
        let mut simple: SampleReturns<StackRingBuffer<2, f64>, 1> =
            SampleReturns::new_stack(ReturnKind::Simple);
        simple.add_lag(1).unwrap();
        simple.update(0.0);
        assert!(!simple.is_ready_at(0));
        simple.update(1.0);
        assert!(simple.is_ready_at(0));
        assert_eq!(simple.value_at(0), None);

        let mut log: SampleReturns<StackRingBuffer<2, f64>, 1> =
            SampleReturns::new_stack(ReturnKind::Log);
        log.add_lag(1).unwrap();
        log.update(-1.0);
        log.update(1.0);
        assert_eq!(log.value_at(0), None);
    }

    #[test]
    fn validates_lags() {
        let mut returns: SampleReturns<StackRingBuffer<2, f64>, 1> =
            SampleReturns::new_stack(ReturnKind::Simple);
        assert!(returns.add_lag(0).is_err());
        assert!(returns.add_lag(2).is_err());
        returns.add_lag(1).unwrap();
        returns.update(1.0);
        assert!(returns.add_lag(1).is_err());
    }
}

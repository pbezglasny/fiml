use std::time::Duration;

use crate::features::BuiltinFeature;
use crate::features::builder::{IndicatorFeatureVectorBuilder, PendingFeature};
use crate::features::event::{Event, EventKind};
use crate::features::vector::{BuiltinFeatureEntry, FeatureKey};
use crate::indicators::{SimpleMovingAverage, SimpleMovingAverageTimed};
use crate::vectors::FeatureOutput;
use crate::{FimlError, Float, HeapRingBuffer, Result, Ticker};

/// Maximum number of SMA windows that can share a single indicator instance.
/// Exceeding it during construction is an error.
pub const MAX_WINDOWS_PER_SMA: usize = 16;

#[derive(Clone, Copy)]
pub(in crate::features) struct PendingSmaPeriods {
    ticker: Ticker,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
    output_start: usize,
}

#[derive(Clone, Copy)]
pub(in crate::features) struct PendingSmaTimedPeriods {
    ticker: Ticker,
    aggregation: Duration,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
    output_start: usize,
}

pub struct SmaFeature<F: Float + 'static> {
    ticker: Ticker,
    sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>,
    output_indexes: [usize; MAX_WINDOWS_PER_SMA],
    output_count: usize,
}

impl<F: Float + 'static> SmaFeature<F> {
    pub(in crate::features) fn new(
        ticker: Ticker,
        sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>,
        output_indexes: [usize; MAX_WINDOWS_PER_SMA],
        output_count: usize,
    ) -> Self {
        Self {
            ticker,
            sma,
            output_indexes,
            output_count,
        }
    }

    pub(in crate::features) fn update<O: FeatureOutput<F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Event::Price(p) = event
            && p.ticker == self.ticker
        {
            self.sma.update(p.value);
            for (window_index, output_index) in self
                .output_indexes
                .iter()
                .enumerate()
                .take(self.output_count)
            {
                if let Some(value) = self.sma.value_at(window_index) {
                    output.set_value_at(*output_index, value);
                }
            }
        }
    }
}

pub struct SmaTimedFeature<F: Float + 'static> {
    ticker: Ticker,
    sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_WINDOWS_PER_SMA>,
    output_indexes: [usize; MAX_WINDOWS_PER_SMA],
    output_count: usize,
}

impl<F: Float + 'static> SmaTimedFeature<F> {
    pub(in crate::features) fn new(
        ticker: Ticker,
        sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_WINDOWS_PER_SMA>,
        output_indexes: [usize; MAX_WINDOWS_PER_SMA],
        output_count: usize,
    ) -> Self {
        Self {
            ticker,
            sma,
            output_indexes,
            output_count,
        }
    }

    pub(in crate::features) fn update<O: FeatureOutput<F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Event::Price(p) = event
            && p.ticker == self.ticker
        {
            self.sma.update_inner(p.value, p.timestamp);
            for (window_index, output_index) in self
                .output_indexes
                .iter()
                .enumerate()
                .take(self.output_count)
            {
                if let Some(value) = self.sma.value_at(window_index) {
                    output.set_value_at(*output_index, value);
                }
            }
        }
    }
}

/// Nested builder for a sample-period SMA indicator.
pub struct SmaPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    ticker: Ticker,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
}

/// Nested builder for a time-bucketed SMA indicator.
pub struct SmaTimedPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    ticker: Ticker,
    aggregation: Duration,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
}

impl<F, V, const M: usize> SmaPeriodsBuilder<F, V, M, false>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    pub(in crate::features) fn new(
        parent: IndicatorFeatureVectorBuilder<F, V, M>,
        ticker: Ticker,
    ) -> Self {
        Self {
            parent,
            ticker,
            periods: [0; MAX_WINDOWS_PER_SMA],
            window_count: 0,
            max_period: 0,
        }
    }

    /// Add the first sample-period SMA window.
    pub fn window(mut self, period: usize) -> Result<SmaPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(SmaPeriodsBuilder {
            parent: self.parent,
            ticker: self.ticker,
            periods: self.periods,
            window_count: self.window_count,
            max_period: self.max_period,
        })
    }
}

impl<F, V, const M: usize> SmaPeriodsBuilder<F, V, M, true>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    /// Add another sample-period SMA window.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the SMA indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::SmaPeriods(PendingSmaPeriods {
                periods: self.periods,
                ticker: self.ticker,
                window_count: self.window_count,
                max_period: self.max_period,
                output_start,
            }));
        Ok(self.parent)
    }
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> SmaPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        validate_period(period)?;
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_SMA, "SMA")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        self.max_period = self.max_period.max(period);
        Ok(())
    }
}

impl<F, V, const M: usize> SmaTimedPeriodsBuilder<F, V, M, false>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    pub(in crate::features) fn new(
        parent: IndicatorFeatureVectorBuilder<F, V, M>,
        ticker: Ticker,
        aggregation: Duration,
    ) -> Self {
        Self {
            parent,
            ticker,
            aggregation,
            periods: [0; MAX_WINDOWS_PER_SMA],
            window_count: 0,
            max_period: 0,
        }
    }

    /// Add the first timed SMA window, measured in aggregation buckets.
    pub fn window(mut self, period: usize) -> Result<SmaTimedPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(SmaTimedPeriodsBuilder {
            parent: self.parent,
            ticker: self.ticker,
            aggregation: self.aggregation,
            periods: self.periods,
            window_count: self.window_count,
            max_period: self.max_period,
        })
    }
}

impl<F, V, const M: usize> SmaTimedPeriodsBuilder<F, V, M, true>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    /// Add another timed SMA window, measured in aggregation buckets.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the timed SMA indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::SmaTimedPeriods(PendingSmaTimedPeriods {
                aggregation: self.aggregation,
                ticker: self.ticker,
                periods: self.periods,
                window_count: self.window_count,
                max_period: self.max_period,
                output_start,
            }));
        Ok(self.parent)
    }
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> SmaTimedPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        if self.aggregation.as_millis() == 0 {
            return Err(FimlError::InvalidArgument(
                "SMA timed aggregation must be at least 1 millisecond".to_string(),
            ));
        }
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                "SMA timed period must be at least 1".to_string(),
            ));
        }
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_SMA, "SMA timed")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        self.max_period = self.max_period.max(period);
        Ok(())
    }
}

pub(in crate::features) fn validate_period(period: usize) -> Result<()> {
    if period == 0 {
        return Err(FimlError::InvalidArgument(
            "SMA period must be at least 1".to_string(),
        ));
    }
    Ok(())
}

pub(in crate::features) fn timed_periods(aggregation: Duration, window: Duration) -> Result<usize> {
    validate_timed_durations(aggregation, window)?;
    Ok((window.as_millis() / aggregation.as_millis()) as usize)
}

pub(in crate::features) fn validate_timed_durations(
    aggregation: Duration,
    window: Duration,
) -> Result<()> {
    let aggregation_millis = aggregation.as_millis();
    let window_millis = window.as_millis();
    if aggregation_millis == 0 {
        return Err(FimlError::InvalidArgument(
            "SMA timed aggregation must be at least 1 millisecond".to_string(),
        ));
    }
    if window_millis < aggregation_millis {
        return Err(FimlError::InvalidArgument(
            "SMA timed window cannot be less than aggregation".to_string(),
        ));
    }
    if !window_millis.is_multiple_of(aggregation_millis) {
        return Err(FimlError::InvalidArgument(
            "SMA timed window must be a multiple of aggregation".to_string(),
        ));
    }
    Ok(())
}

pub(in crate::features) fn build_builtin<F: Float + 'static>(
    ticker: Ticker,
    period: usize,
    output_index: usize,
) -> Result<BuiltinFeature<F>> {
    let mut sma =
        SimpleMovingAverage::<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>::new_heap(period);
    sma.add_window(period)
        .expect("validated SMA period should fit its ring buffer");
    let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];
    output_indexes[0] = output_index;
    Ok(BuiltinFeature::Sma(SmaFeature::new(
        ticker,
        sma,
        output_indexes,
        1,
    )))
}

pub(in crate::features) fn build_timed_builtin<F: Float + 'static>(
    ticker: Ticker,
    aggregation: Duration,
    window: Duration,
    output_index: usize,
) -> Result<BuiltinFeature<F>> {
    let period = timed_periods(aggregation, window)?;
    let capacity = period
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("SMA timed period is too large".to_string()))?;
    let mut sma =
        SimpleMovingAverageTimed::<HeapRingBuffer<(i64, F)>, F, MAX_WINDOWS_PER_SMA>::new_heap(
            aggregation,
            capacity,
        )?;
    sma.add_window_with_periods(period)
        .expect("validated SMA timed period should fit its ring buffer");
    let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];
    output_indexes[0] = output_index;
    Ok(BuiltinFeature::SmaTimed(SmaTimedFeature::new(
        ticker,
        sma,
        output_indexes,
        1,
    )))
}

pub(in crate::features) fn build_sma_periods_entry<F: Float + 'static>(
    config: &PendingSmaPeriods,
    names: &mut [Option<FeatureKey>],
) -> BuiltinFeatureEntry<F> {
    let mut sma = SimpleMovingAverage::<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>::new_heap(
        config.max_period,
    );
    let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];

    for (window_index, period) in config
        .periods
        .iter()
        .copied()
        .enumerate()
        .take(config.window_count)
    {
        sma.add_window(period)
            .expect("validated SMA period should fit its ring buffer");
        let output_index = config.output_start + window_index;
        output_indexes[window_index] = output_index;
        names[output_index] = Some(FeatureKey {
            ticker: config.ticker,
            name: format!("sma_periods_{period}"),
        });
    }

    BuiltinFeatureEntry {
        feature: BuiltinFeature::Sma(SmaFeature::new(
            config.ticker,
            sma,
            output_indexes,
            config.window_count,
        )),
        kind: EventKind::Price,
    }
}

pub(in crate::features) fn build_sma_timed_periods_entry<F: Float + 'static>(
    config: &PendingSmaTimedPeriods,
    names: &mut [Option<FeatureKey>],
) -> Result<BuiltinFeatureEntry<F>> {
    let capacity = config
        .max_period
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("SMA timed period is too large".to_string()))?;
    let mut sma =
        SimpleMovingAverageTimed::<HeapRingBuffer<(i64, F)>, F, MAX_WINDOWS_PER_SMA>::new_heap(
            config.aggregation,
            capacity,
        )?;
    let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];

    for (window_index, period) in config
        .periods
        .iter()
        .copied()
        .enumerate()
        .take(config.window_count)
    {
        sma.add_window_with_periods(period)
            .expect("validated SMA timed period should fit its ring buffer");
        let output_index = config.output_start + window_index;
        output_indexes[window_index] = output_index;
        names[output_index] = Some(FeatureKey {
            ticker: config.ticker,
            name: format!("sma_timed_periods_{period}"),
        });
    }

    Ok(BuiltinFeatureEntry {
        feature: BuiltinFeature::SmaTimed(SmaTimedFeature::new(
            config.ticker,
            sma,
            output_indexes,
            config.window_count,
        )),
        kind: EventKind::Price,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, ticker};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn sma_reacts_to_price_events() {
        let aapl = ticker::intern("AAPL");
        let googl = ticker::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, MAX_WINDOWS_PER_SMA> =
            SimpleMovingAverage::new_heap(3);
        sma.add_window(3).unwrap();
        let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];
        output_indexes[0] = 0;

        let mut feat = SmaFeature::new(aapl, sma, output_indexes, 1);
        for v in [3.0, 6.0, 9.0] {
            feat.update(&Event::price(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::price(googl, 30.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 6.0));
    }
}

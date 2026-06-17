use std::time::Duration;

use crate::features::BuiltinFeature;
use crate::features::event::{Event, EventKind, market_value_for_kind};
use crate::features::indicator_vector::{BuiltinFeatureEntry, FeatureKey};
use crate::indicators::{
    PendingSmaPeriods, PendingSmaTimedPeriods, SimpleMovingAverage, SimpleMovingAverageTimed,
};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, HeapRingBuffer, Result, Symbol};

/// Maximum number of SMA windows that can share a single indicator instance.
/// Exceeding it during construction is an error.
pub const MAX_WINDOWS_PER_SMA: usize = 16;

pub struct SmaFeature<F: Float> {
    ticker: Symbol,
    event_kind: EventKind,
    sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>,
    output_indexes: [usize; MAX_WINDOWS_PER_SMA],
    output_count: usize,
}

impl<F: Float> SmaFeature<F> {
    pub(crate) fn new(
        ticker: Symbol,
        event_kind: EventKind,
        sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>,
        output_indexes: [usize; MAX_WINDOWS_PER_SMA],
        output_count: usize,
    ) -> Self {
        Self {
            ticker,
            event_kind,
            sma,
            output_indexes,
            output_count,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Some(value) = market_value_for_kind(event, self.event_kind, self.ticker) {
            self.sma.update(value);
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

pub struct SmaTimedFeature<F: Float> {
    ticker: Symbol,
    sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_WINDOWS_PER_SMA>,
    output_indexes: [usize; MAX_WINDOWS_PER_SMA],
    output_count: usize,
}

impl<F: Float> SmaTimedFeature<F> {
    pub(crate) fn new(
        ticker: Symbol,
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

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
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

pub(crate) fn validate_period(period: usize) -> Result<()> {
    if period == 0 {
        return Err(FimlError::InvalidArgument(
            "SMA period must be at least 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_event_kind(event_kind: EventKind) -> Result<()> {
    match event_kind {
        EventKind::Price | EventKind::Volume => Ok(()),
        EventKind::OrderBook | EventKind::Time => Err(FimlError::InvalidArgument(
            "SMA event kind must be price or volume".to_string(),
        )),
    }
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

pub(in crate::features) fn build_builtin<F: Float>(
    ticker: Symbol,
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
        EventKind::Price,
        sma,
        output_indexes,
        1,
    )))
}

pub(in crate::features) fn build_timed_builtin<F: Float>(
    ticker: Symbol,
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

pub(crate) fn build_sma_periods_entry<F: Float>(
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
            name: feature_name(config.event_kind, period),
        });
    }

    BuiltinFeatureEntry {
        feature: BuiltinFeature::Sma(SmaFeature::new(
            config.ticker,
            config.event_kind,
            sma,
            output_indexes,
            config.window_count,
        )),
        kind: config.event_kind,
    }
}

fn feature_name(event_kind: EventKind, period: usize) -> String {
    match event_kind {
        EventKind::Price => format!("sma_periods_{period}"),
        EventKind::Volume => format!("volume_sma_periods_{period}"),
        EventKind::OrderBook | EventKind::Time => {
            unreachable!("validated SMA event kind should be price or volume")
        }
    }
}

pub(crate) fn build_sma_timed_periods_entry<F: Float>(
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

        let mut feat = SmaFeature::new(aapl, EventKind::Price, sma, output_indexes, 1);
        for v in [3.0, 6.0, 9.0] {
            feat.update(&Event::price(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::volume(aapl, 90.0, 0), &mut fv);
        feat.update(&Event::price(googl, 30.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 6.0));
    }

    #[test]
    fn sma_reacts_to_volume_events() {
        let aapl = ticker::intern("AAPL");
        let googl = ticker::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, MAX_WINDOWS_PER_SMA> =
            SimpleMovingAverage::new_heap(3);
        sma.add_window(3).unwrap();
        let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];
        output_indexes[0] = 0;

        let mut feat = SmaFeature::new(aapl, EventKind::Volume, sma, output_indexes, 1);
        feat.update(&Event::price(aapl, 1_000.0, 0), &mut fv);
        for v in [100.0, 200.0, 300.0] {
            feat.update(&Event::volume(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::volume(googl, 3_000.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 200.0));
    }
}

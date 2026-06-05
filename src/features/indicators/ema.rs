use crate::features::BuiltinFeature;
use crate::features::builder::{IndicatorFeatureVectorBuilder, PendingFeature};
use crate::features::event::Event;
use crate::features::event::EventKind;
use crate::features::vector::{BuiltinFeatureEntry, FeatureKey};
use crate::indicators::ExponentialMovingAverage;
use crate::vectors::FeatureOutput;
use crate::{FimlError, Float, Result, Ticker};

pub const MAX_WINDOWS_PER_EMA: usize = super::sma::MAX_WINDOWS_PER_SMA;

#[derive(Clone, Copy)]
pub(in crate::features) struct PendingEmaPeriods {
    ticker: Ticker,
    periods: [usize; MAX_WINDOWS_PER_EMA],
    window_count: usize,
    output_start: usize,
}

pub struct EmaFeature<F: Float + 'static> {
    ticker: Ticker,
    ema: ExponentialMovingAverage<F, MAX_WINDOWS_PER_EMA>,
    output_indexes: [usize; MAX_WINDOWS_PER_EMA],
    output_count: usize,
}

impl<F: Float + 'static> EmaFeature<F> {
    pub(in crate::features) fn new(
        ticker: Ticker,
        ema: ExponentialMovingAverage<F, MAX_WINDOWS_PER_EMA>,
        output_indexes: [usize; MAX_WINDOWS_PER_EMA],
        output_count: usize,
    ) -> Self {
        Self {
            ticker,
            ema,
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
            self.ema.update(p.value);
            for (window_index, output_index) in self
                .output_indexes
                .iter()
                .enumerate()
                .take(self.output_count)
            {
                if let Some(value) = self.ema.value_at(window_index) {
                    output.set_value_at(*output_index, value);
                }
            }
        }
    }
}

/// Nested builder for a sample-period EMA indicator.
pub struct EmaPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    ticker: Ticker,
    periods: [usize; MAX_WINDOWS_PER_EMA],
    window_count: usize,
}

impl<F, V, const M: usize> EmaPeriodsBuilder<F, V, M, false>
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
            periods: [0; MAX_WINDOWS_PER_EMA],
            window_count: 0,
        }
    }

    /// Add the first sample-period EMA window.
    pub fn window(mut self, period: usize) -> Result<EmaPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(EmaPeriodsBuilder {
            parent: self.parent,
            ticker: self.ticker,
            periods: self.periods,
            window_count: self.window_count,
        })
    }
}

impl<F, V, const M: usize> EmaPeriodsBuilder<F, V, M, true>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    /// Add another sample-period EMA window.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the EMA indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::EmaPeriods(PendingEmaPeriods {
                periods: self.periods,
                ticker: self.ticker,
                window_count: self.window_count,
                output_start,
            }));
        Ok(self.parent)
    }
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> EmaPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        validate_period(period)?;
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_EMA, "EMA")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        Ok(())
    }
}

pub(in crate::features) fn validate_period(period: usize) -> Result<()> {
    if period == 0 {
        return Err(FimlError::InvalidArgument(
            "EMA period must be at least 1".to_string(),
        ));
    }
    Ok(())
}

pub(in crate::features) fn build_builtin<F: Float + 'static>(
    ticker: Ticker,
    period: usize,
    output_index: usize,
) -> Result<BuiltinFeature<F>> {
    let mut ema = ExponentialMovingAverage::<F, MAX_WINDOWS_PER_EMA>::new();
    ema.add_window(period)
        .expect("validated EMA period should fit its window storage");
    let mut output_indexes = [0; MAX_WINDOWS_PER_EMA];
    output_indexes[0] = output_index;
    Ok(BuiltinFeature::Ema(EmaFeature::new(
        ticker,
        ema,
        output_indexes,
        1,
    )))
}

pub(in crate::features) fn build_ema_periods_entry<F: Float + 'static>(
    config: &PendingEmaPeriods,
    names: &mut [Option<FeatureKey>],
) -> BuiltinFeatureEntry<F> {
    let mut ema = ExponentialMovingAverage::<F, MAX_WINDOWS_PER_EMA>::new();
    let mut output_indexes = [0; MAX_WINDOWS_PER_EMA];

    for (window_index, period) in config
        .periods
        .iter()
        .copied()
        .enumerate()
        .take(config.window_count)
    {
        ema.add_window(period)
            .expect("validated EMA period should fit its window storage");
        let output_index = config.output_start + window_index;
        output_indexes[window_index] = output_index;
        names[output_index] = Some(FeatureKey {
            ticker: config.ticker,
            name: format!("ema_periods_{period}"),
        });
    }

    BuiltinFeatureEntry {
        feature: BuiltinFeature::Ema(EmaFeature::new(
            config.ticker,
            ema,
            output_indexes,
            config.window_count,
        )),
        kind: EventKind::Price,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, ticker};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn ema_reacts_to_price_events() {
        let aapl = ticker::intern("AAPL");
        let googl = ticker::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut ema: ExponentialMovingAverage<f64, MAX_WINDOWS_PER_EMA> =
            ExponentialMovingAverage::new();
        ema.add_window(3).unwrap();
        let mut output_indexes = [0; MAX_WINDOWS_PER_EMA];
        output_indexes[0] = 0;

        let mut feat = EmaFeature::new(aapl, ema, output_indexes, 1);
        for v in [10.0, 20.0, 30.0] {
            feat.update(&Event::price(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::price(googl, 300.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 22.5));
    }
}

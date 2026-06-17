use crate::features::BuiltinFeature;
use crate::features::event::Event;
use crate::features::event::EventKind;
use crate::features::event::market_value_for_kind;
use crate::features::indicator_vector::{BuiltinFeatureEntry, FeatureKey};
use crate::indicators::{ExponentialMovingAverage, PendingEmaPeriods};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, Result, Symbol};

pub const MAX_WINDOWS_PER_EMA: usize = super::sma::MAX_WINDOWS_PER_SMA;

pub struct EmaFeature<F: Float> {
    ticker: Symbol,
    event_kind: EventKind,
    ema: ExponentialMovingAverage<F, MAX_WINDOWS_PER_EMA>,
    output_indexes: [usize; MAX_WINDOWS_PER_EMA],
    output_count: usize,
}

impl<F: Float> EmaFeature<F> {
    pub(crate) fn new(
        ticker: Symbol,
        event_kind: EventKind,
        ema: ExponentialMovingAverage<F, MAX_WINDOWS_PER_EMA>,
        output_indexes: [usize; MAX_WINDOWS_PER_EMA],
        output_count: usize,
    ) -> Self {
        Self {
            ticker,
            event_kind,
            ema,
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
            self.ema.update(value);
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

pub(crate) fn validate_period(period: usize) -> Result<()> {
    if period == 0 {
        return Err(FimlError::InvalidArgument(
            "EMA period must be at least 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_event_kind(event_kind: EventKind) -> Result<()> {
    match event_kind {
        EventKind::Price | EventKind::Volume => Ok(()),
        EventKind::OrderBook | EventKind::Time => Err(FimlError::InvalidArgument(
            "EMA event kind must be price or volume".to_string(),
        )),
    }
}

pub(in crate::features) fn build_builtin<F: Float>(
    ticker: Symbol,
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
        EventKind::Price,
        ema,
        output_indexes,
        1,
    )))
}

pub(crate) fn build_ema_periods_entry<F: Float>(
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
            name: feature_name(config.event_kind, period),
        });
    }

    BuiltinFeatureEntry {
        feature: BuiltinFeature::Ema(EmaFeature::new(
            config.ticker,
            config.event_kind,
            ema,
            output_indexes,
            config.window_count,
        )),
        kind: config.event_kind,
    }
}

fn feature_name(event_kind: EventKind, period: usize) -> String {
    match event_kind {
        EventKind::Price => format!("ema_periods_{period}"),
        EventKind::Volume => format!("volume_ema_periods_{period}"),
        EventKind::OrderBook | EventKind::Time => {
            unreachable!("validated EMA event kind should be price or volume")
        }
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

        let mut feat = EmaFeature::new(aapl, EventKind::Price, ema, output_indexes, 1);
        for v in [10.0, 20.0, 30.0] {
            feat.update(&Event::price(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::volume(aapl, 300.0, 0), &mut fv);
        feat.update(&Event::price(googl, 300.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 22.5));
    }

    #[test]
    fn ema_reacts_to_volume_events() {
        let aapl = ticker::intern("AAPL");
        let googl = ticker::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut ema: ExponentialMovingAverage<f64, MAX_WINDOWS_PER_EMA> =
            ExponentialMovingAverage::new();
        ema.add_window(3).unwrap();
        let mut output_indexes = [0; MAX_WINDOWS_PER_EMA];
        output_indexes[0] = 0;

        let mut feat = EmaFeature::new(aapl, EventKind::Volume, ema, output_indexes, 1);
        feat.update(&Event::price(aapl, 1_000.0, 0), &mut fv);
        for v in [100.0, 200.0, 300.0] {
            feat.update(&Event::volume(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::volume(googl, 3_000.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 225.0));
    }
}

use crate::features::BuiltinFeature;
use crate::features::builtin::write_outputs;
use crate::features::compiler::OutputSpan;
use crate::features::definition::{MAX_OUTPUTS_PER_INDICATOR, ValueSource};
use crate::features::event::Event;
use crate::indicators::ExponentialMovingAverage;
use crate::vectors::FeatureVector;
use crate::{Float, Result, Symbol};

pub struct EmaFeature<F: Float> {
    symbol: Symbol,
    source: ValueSource,
    ema: ExponentialMovingAverage<F, MAX_OUTPUTS_PER_INDICATOR>,
    output_span: OutputSpan,
}

impl<F: Float> EmaFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        source: ValueSource,
        ema: ExponentialMovingAverage<F, MAX_OUTPUTS_PER_INDICATOR>,
        output_span: OutputSpan,
    ) -> Self {
        Self {
            symbol,
            source,
            ema,
            output_span,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Some(value) = self.source.value(event, self.symbol) {
            self.ema.update(value);
            write_outputs(self.output_span, output, |index| self.ema.value_at(index));
        }
    }
}

pub(crate) fn build<F: Float>(
    symbol: Symbol,
    source: ValueSource,
    windows: &[usize],
    output_span: OutputSpan,
) -> Result<BuiltinFeature<F>> {
    debug_assert_eq!(windows.len(), output_span.count);
    let mut ema = ExponentialMovingAverage::<F, MAX_OUTPUTS_PER_INDICATOR>::new();
    for &window in windows {
        ema.add_window(window)?;
    }
    Ok(BuiltinFeature::Ema(EmaFeature::new(
        symbol,
        source,
        ema,
        output_span,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn ema_reacts_to_price_events() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut ema: ExponentialMovingAverage<f64, MAX_OUTPUTS_PER_INDICATOR> =
            ExponentialMovingAverage::new();
        ema.add_window(3).unwrap();

        let mut feat = EmaFeature::new(
            aapl,
            ValueSource::Price,
            ema,
            OutputSpan { start: 0, count: 1 },
        );
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
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut ema: ExponentialMovingAverage<f64, MAX_OUTPUTS_PER_INDICATOR> =
            ExponentialMovingAverage::new();
        ema.add_window(3).unwrap();

        let mut feat = EmaFeature::new(
            aapl,
            ValueSource::Volume,
            ema,
            OutputSpan { start: 0, count: 1 },
        );
        feat.update(&Event::price(aapl, 1_000.0, 0), &mut fv);
        for v in [100.0, 200.0, 300.0] {
            feat.update(&Event::volume(aapl, v, 0), &mut fv);
        }
        feat.update(&Event::volume(googl, 3_000.0, 0), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 225.0));
    }

    #[test]
    fn ema_reacts_to_trade_price_events() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut ema: ExponentialMovingAverage<f64, MAX_OUTPUTS_PER_INDICATOR> =
            ExponentialMovingAverage::new();
        ema.add_window(3).unwrap();

        let mut feat = EmaFeature::new(
            aapl,
            ValueSource::TradePrice,
            ema,
            OutputSpan { start: 0, count: 1 },
        );
        feat.update(&Event::price(aapl, 1_000.0, 0), &mut fv);
        feat.update(&Event::volume(aapl, 1_000.0, 0), &mut fv);
        for price in [10.0, 20.0, 30.0] {
            feat.update(&Event::trade(aapl, price, 100.0, 0, None), &mut fv);
        }
        feat.update(&Event::trade(googl, 300.0, 100.0, 0, None), &mut fv);
        feat.update(&Event::time(123), &mut fv);

        assert!(approx_eq(fv.values()[0], 22.5));
    }
}

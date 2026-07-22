use std::time::Duration;

use crate::features::BuiltinFeature;
use crate::features::builtin::write_outputs;
use crate::features::compiler::OutputSpan;
use crate::features::definition::{MAX_OUTPUTS_PER_INDICATOR, ValueSource};
use crate::features::event::Event;
use crate::indicators::{SimpleMovingAverage, SimpleMovingAverageTimed};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, HeapRingBuffer, Result, Symbol};

pub struct SmaFeature<F: Float> {
    symbol: Symbol,
    source: ValueSource,
    sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
    output_span: OutputSpan,
}

impl<F: Float> SmaFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        source: ValueSource,
        sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
        output_span: OutputSpan,
    ) -> Self {
        Self {
            symbol,
            source,
            sma,
            output_span,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Some(value) = self.source.value(event, self.symbol) {
            self.sma.update(value);
            write_outputs(self.output_span, output, |index| self.sma.value_at(index));
        }
    }
}

pub struct SmaTimedFeature<F: Float> {
    symbol: Symbol,
    source: ValueSource,
    sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_OUTPUTS_PER_INDICATOR>,
    output_span: OutputSpan,
}

impl<F: Float> SmaTimedFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        source: ValueSource,
        sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_OUTPUTS_PER_INDICATOR>,
        output_span: OutputSpan,
    ) -> Self {
        Self {
            symbol,
            source,
            sma,
            output_span,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Some(value) = self.source.value(event, self.symbol) {
            self.sma.update_inner(value, event.timestamp());
            write_outputs(self.output_span, output, |index| self.sma.value_at(index));
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
    let max_window = windows.iter().copied().max().unwrap_or(0);
    let mut sma = SimpleMovingAverage::<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>::new_heap(
        max_window,
    );
    for &window in windows {
        sma.add_window(window)?;
    }
    Ok(BuiltinFeature::Sma(SmaFeature::new(
        symbol,
        source,
        sma,
        output_span,
    )))
}

pub(crate) fn build_timed<F: Float>(
    symbol: Symbol,
    source: ValueSource,
    aggregation: Duration,
    periods: &[usize],
    max_period: usize,
    output_span: OutputSpan,
) -> Result<BuiltinFeature<F>> {
    debug_assert_eq!(periods.len(), output_span.count);
    let capacity = max_period
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("SMA timed period is too large".to_string()))?;
    let mut sma = SimpleMovingAverageTimed::<
        HeapRingBuffer<(i64, F)>,
        F,
        MAX_OUTPUTS_PER_INDICATOR,
    >::new_heap(aggregation, capacity)?;
    for &period in periods {
        sma.add_window_with_periods(period)?;
    }
    Ok(BuiltinFeature::SmaTimed(SmaTimedFeature::new(
        symbol,
        source,
        sma,
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
    fn grouped_sma_writes_adjacent_outputs() {
        let symbol = symbols::intern("AAPL");
        let mut feature = match build::<f64>(
            symbol,
            ValueSource::Price,
            &[2, 3],
            OutputSpan { start: 0, count: 2 },
        )
        .unwrap()
        {
            BuiltinFeature::Sma(feature) => feature,
            _ => unreachable!(),
        };
        let mut output = ArrayFeatureVector::<f64, 2>::new();

        for value in [1.0, 2.0, 3.0] {
            feature.update(&Event::price(symbol, value, 0), &mut output);
        }

        assert!(approx_eq(output.values()[0], 2.5));
        assert!(approx_eq(output.values()[1], 2.0));
    }

    #[test]
    fn sma_can_consume_trade_volume() {
        let symbol = symbols::intern("AAPL");
        let mut feature = match build::<f64>(
            symbol,
            ValueSource::TradeVolume,
            &[2],
            OutputSpan { start: 0, count: 1 },
        )
        .unwrap()
        {
            BuiltinFeature::Sma(feature) => feature,
            _ => unreachable!(),
        };
        let mut output = ArrayFeatureVector::<f64, 1>::new();

        feature.update(&Event::trade(symbol, 100.0, 4.0, 0, None), &mut output);
        feature.update(&Event::trade(symbol, 101.0, 6.0, 1, None), &mut output);

        assert!(approx_eq(output.values()[0], 5.0));
    }
}

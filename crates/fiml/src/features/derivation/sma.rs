use std::time::Duration;

use crate::event::Event;
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputSpan;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::{SimpleMovingAverage, SimpleMovingAverageTimed};
use crate::vectors::FeatureVector;
use crate::{
    EventField, FimlError, Float, HeapRingBuffer, IndicatorKind, InvalidArgumentError, Result,
    Symbol, WarmupPolicy,
};

pub(crate) struct SmaFeature<F: Float> {
    symbol: Symbol,
    source: EventField,
    sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
}

impl<F: Float> SmaFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        source: EventField,
        sma: SimpleMovingAverage<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self {
            symbol,
            source,
            sma,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output_span: OutputSpan,
        output: &mut O,
    ) {
        if event.symbol() == self.symbol
            && let Some(value) = self.source.extract(event)
        {
            self.sma.update(value);
            write_outputs(output_span, output, |index| self.sma.value_at(index));
        }
    }
}

pub(crate) struct SmaTimedFeature<F: Float> {
    symbol: Symbol,
    source: EventField,
    sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_OUTPUTS_PER_INDICATOR>,
}

impl<F: Float> SmaTimedFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        source: EventField,
        sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self {
            symbol,
            source,
            sma,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output_span: OutputSpan,
        output: &mut O,
    ) {
        if event.symbol() == self.symbol
            && let Some(value) = self.source.extract(event)
        {
            self.sma.update(value, event.timestamp());
        } else if !self.sma.observe(event.timestamp()) {
            return;
        }

        write_outputs(output_span, output, |index| self.sma.value_at(index));
    }
}

pub(crate) fn build<F: Float>(
    symbol: Symbol,
    source: EventField,
    windows: &[usize],
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation<F>> {
    let max_window = windows.iter().copied().max().unwrap_or(0);
    let mut sma = SimpleMovingAverage::<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>::new_heap(
        max_window,
        warmup_policy,
    );
    for &window in windows {
        sma.add_window(window)?;
    }
    Ok(FeatureDerivation::Sma(SmaFeature::new(symbol, source, sma)))
}

pub(crate) fn build_timed<F: Float>(
    symbol: Symbol,
    source: EventField,
    aggregation: Duration,
    periods: &[usize],
    max_period: usize,
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation<F>> {
    let capacity = max_period.checked_add(1).ok_or(FimlError::InvalidArgument(
        InvalidArgumentError::TimedPeriodTooLarge {
            indicator: IndicatorKind::SmaTimed,
        },
    ))?;
    let mut sma = SimpleMovingAverageTimed::<
        HeapRingBuffer<(i64, F)>,
        F,
        MAX_OUTPUTS_PER_INDICATOR,
    >::new_heap(aggregation, capacity, warmup_policy)?;
    for &period in periods {
        sma.add_window_with_periods(period)?;
    }
    Ok(FeatureDerivation::SmaTimed(SmaTimedFeature::new(
        symbol, source, sma,
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
        let mut feature =
            match build::<f64>(symbol, EventField::Price, &[2, 3], WarmupPolicy::FullWindow)
                .unwrap()
            {
                FeatureDerivation::Sma(feature) => feature,
                _ => unreachable!(),
            };
        let mut output = ArrayFeatureVector::<f64, 2>::new();
        let output_span = OutputSpan { start: 0, count: 2 };

        for value in [1.0, 2.0, 3.0] {
            feature.update(&Event::price(symbol, value, 0), output_span, &mut output);
        }

        assert!(approx_eq(output.values()[0], 2.5));
        assert!(approx_eq(output.values()[1], 2.0));
    }

    #[test]
    fn sma_can_consume_trade_volume() {
        let symbol = symbols::intern("AAPL");
        let mut feature = match build::<f64>(
            symbol,
            EventField::TradeVolume,
            &[2],
            WarmupPolicy::FullWindow,
        )
        .unwrap()
        {
            FeatureDerivation::Sma(feature) => feature,
            _ => unreachable!(),
        };
        let mut output = ArrayFeatureVector::<f64, 1>::new();
        let output_span = OutputSpan { start: 0, count: 1 };

        feature.update(
            &Event::trade(symbol, 100.0, 4.0, 0, None),
            output_span,
            &mut output,
        );
        feature.update(
            &Event::trade(symbol, 101.0, 6.0, 1, None),
            output_span,
            &mut output,
        );

        assert!(approx_eq(output.values()[0], 5.0));
    }
}

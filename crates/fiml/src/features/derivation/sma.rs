//! Adapts timed simple moving averages to compiled feature outputs.
//!
use std::time::Duration;

use crate::event::Event;
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputRange;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::SimpleMovingAverageTimed;
use crate::vectors::FeatureVector;
use crate::{
    EventField, FimlError, HeapRingBuffer, IndicatorKind, InvalidArgumentError, Result, Symbol,
    WarmupPolicy,
};

pub(crate) struct SmaTimedFeature {
    symbol: Symbol,
    source: EventField,
    sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, MAX_OUTPUTS_PER_INDICATOR>,
}

impl SmaTimedFeature {
    pub(crate) fn new(
        symbol: Symbol,
        source: EventField,
        sma: SimpleMovingAverageTimed<HeapRingBuffer<(i64, f64)>, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self {
            symbol,
            source,
            sma,
        }
    }

    pub(crate) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_range: OutputRange,
        output: &mut O,
    ) {
        if event.symbol() != self.symbol {
            return;
        }
        if let Some(value) = self.source.extract(event) {
            self.sma.update(value, event.timestamp());
        } else if !self.sma.observe(event.timestamp()) {
            return;
        }

        write_outputs(output_range, output, |index| self.sma.value_at(index));
    }
}

pub(crate) fn build_timed(
    symbol: Symbol,
    source: EventField,
    aggregation: Duration,
    periods: &[usize],
    max_period: usize,
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation> {
    let capacity = max_period.checked_add(1).ok_or(FimlError::InvalidArgument(
        InvalidArgumentError::TimedPeriodTooLarge {
            indicator: IndicatorKind::SmaTimed,
        },
    ))?;
    let mut sma = SimpleMovingAverageTimed::<
        HeapRingBuffer<(i64, f64)>,
        MAX_OUTPUTS_PER_INDICATOR,
    >::new_heap(aggregation, capacity, warmup_policy)?;
    for &period in periods {
        sma.add_window_with_periods(period)?;
    }
    Ok(FeatureDerivation::SmaTimed(SmaTimedFeature::new(
        symbol, source, sma,
    )))
}

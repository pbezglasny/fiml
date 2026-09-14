//! Adapts sample and timed rolling volatility to compiled feature outputs.
//!
use std::time::Duration;

use crate::event::Event;
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputRange;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::{RollingVolatility, RollingVolatilityTimed, VolatilityBucket};
use crate::vectors::FeatureVector;
use crate::{
    EventField, FimlError, HeapRingBuffer, IndicatorKind, InvalidArgumentError, Result, Symbol,
    WarmupPolicy,
};

/// Adapts sample-window rolling volatility to feature-vector outputs.
pub(crate) struct VolatilityFeature {
    symbol: Symbol,
    source: EventField,
    volatility: RollingVolatility<HeapRingBuffer<f64>, MAX_OUTPUTS_PER_INDICATOR>,
}

impl VolatilityFeature {
    fn new(
        symbol: Symbol,
        source: EventField,
        volatility: RollingVolatility<HeapRingBuffer<f64>, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self {
            symbol,
            source,
            volatility,
        }
    }

    pub(crate) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_range: OutputRange,
        output: &mut O,
    ) {
        if event.symbol() == self.symbol
            && let Some(value) = self.source.extract(event)
        {
            self.volatility.update(value);
            write_outputs(output_range, output, |index| {
                self.volatility.value_at(index)
            });
        }
    }
}

/// Adapts timed rolling volatility to feature-vector outputs.
pub(crate) struct VolatilityTimedFeature {
    symbol: Symbol,
    source: EventField,
    volatility: RollingVolatilityTimed<HeapRingBuffer<VolatilityBucket>, MAX_OUTPUTS_PER_INDICATOR>,
}

impl VolatilityTimedFeature {
    fn new(
        symbol: Symbol,
        source: EventField,
        volatility: RollingVolatilityTimed<
            HeapRingBuffer<VolatilityBucket>,
            MAX_OUTPUTS_PER_INDICATOR,
        >,
    ) -> Self {
        Self {
            symbol,
            source,
            volatility,
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
            self.volatility.update(value, event.timestamp());
        } else if !self.volatility.observe(event.timestamp()) {
            return;
        }
        write_outputs(output_range, output, |index| {
            self.volatility.value_at(index)
        });
    }
}

pub(crate) fn build(
    symbol: Symbol,
    source: EventField,
    windows: &[usize],
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation> {
    let mut volatility =
        RollingVolatility::new_heap(windows.iter().copied().max().unwrap_or(0), warmup_policy);
    for &window in windows {
        volatility.add_window(window)?;
    }
    Ok(FeatureDerivation::Volatility(VolatilityFeature::new(
        symbol, source, volatility,
    )))
}

pub(crate) fn build_timed(
    symbol: Symbol,
    source: EventField,
    aggregation: Duration,
    periods: &[usize],
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation> {
    let max_period = periods.iter().copied().max().unwrap_or(0);
    let capacity = max_period.checked_add(1).ok_or(FimlError::InvalidArgument(
        InvalidArgumentError::TimedPeriodTooLarge {
            indicator: IndicatorKind::VolatilityTimed,
        },
    ))?;
    let mut volatility = RollingVolatilityTimed::new_heap(aggregation, capacity, warmup_policy)?;
    for &period in periods {
        volatility.add_window_with_periods(period)?;
    }
    Ok(FeatureDerivation::VolatilityTimed(
        VolatilityTimedFeature::new(symbol, source, volatility),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, Symbol};

    #[test]
    fn grouped_volatility_writes_adjacent_outputs() {
        let symbol = Symbol::new("AAPL").unwrap();
        let mut feature =
            match build(symbol, EventField::Price, &[2, 3], WarmupPolicy::FirstValue).unwrap() {
                FeatureDerivation::Volatility(feature) => feature,
                _ => unreachable!(),
            };
        let mut output = ArrayFeatureVector::<2>::new();
        let range = OutputRange { start: 0, count: 2 };

        for value in [100.0, 110.0, 99.0, 118.8] {
            feature.update(&Event::price(symbol, value, 0), range, &mut output);
        }

        assert!((output.values()[0] - 0.15).abs() < 1e-12);
        assert!((output.values()[1] - 0.1247219128924647).abs() < 1e-12);
    }
}

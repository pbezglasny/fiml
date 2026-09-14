//! Adapts timed volume-weighted average price to compiled feature outputs.
//!
use std::time::Duration;

use crate::event::Event;
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputRange;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::{VolumeWeightedAveragePriceTimed, VwapBucket};
use crate::vectors::FeatureVector;
use crate::{
    FimlError, HeapRingBuffer, IndicatorKind, InvalidArgumentError, Result, Symbol, WarmupPolicy,
};

/// Adapts grouped timed VWAP calculations to feature-vector outputs.
pub(crate) struct VwapTimedFeature {
    symbol: Symbol,
    vwap: VolumeWeightedAveragePriceTimed<HeapRingBuffer<VwapBucket>, MAX_OUTPUTS_PER_INDICATOR>,
}

impl VwapTimedFeature {
    fn new(
        symbol: Symbol,
        vwap: VolumeWeightedAveragePriceTimed<
            HeapRingBuffer<VwapBucket>,
            MAX_OUTPUTS_PER_INDICATOR,
        >,
    ) -> Self {
        Self { symbol, vwap }
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
        if let Event::Trade(trade) = event {
            self.vwap
                .update_inner(trade.price, trade.volume, trade.timestamp);
        } else if !self.vwap.observe(event.timestamp()) {
            return;
        }
        write_outputs(output_range, output, |index| self.vwap.window_value(index));
    }
}

pub(crate) fn build_timed(
    symbol: Symbol,
    aggregation: Duration,
    periods: &[usize],
    max_period: usize,
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation> {
    let capacity = max_period.checked_add(1).ok_or(FimlError::InvalidArgument(
        InvalidArgumentError::TimedPeriodTooLarge {
            indicator: IndicatorKind::VwapTimed,
        },
    ))?;
    let mut vwap = VolumeWeightedAveragePriceTimed::<
        HeapRingBuffer<VwapBucket>,
        MAX_OUTPUTS_PER_INDICATOR,
    >::new_heap(aggregation, capacity, warmup_policy)?;
    for &period in periods {
        vwap.add_window_with_periods(period)?;
    }
    Ok(FeatureDerivation::VwapTimed(VwapTimedFeature::new(
        symbol, vwap,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    #[test]
    fn consumes_matching_trades_and_observes_other_events() {
        let aapl = symbols::intern("AAPL").unwrap();
        let mut output = ArrayFeatureVector::<2>::new();
        let mut feature = build_timed(
            aapl,
            Duration::from_secs(1),
            &[2, 3],
            3,
            WarmupPolicy::FirstValue,
        )
        .unwrap();
        let FeatureDerivation::VwapTimed(feature) = &mut feature else {
            unreachable!()
        };
        let range = OutputRange { start: 0, count: 2 };
        feature.update(&Event::trade(aapl, 10.0, 1.0, 0, None), range, &mut output);
        feature.update(
            &Event::trade(aapl, 20.0, 3.0, 1_000, None),
            range,
            &mut output,
        );
        feature.update(&Event::price(aapl, 12.0, 2_000), range, &mut output);
        assert_eq!(output.values(), &[20.0, 17.5]);
    }
}

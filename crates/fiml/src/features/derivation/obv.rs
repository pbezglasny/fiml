use std::time::Duration;

use crate::event::Event;
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputSpan;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::{ObvBucket, OnBalanceVolumeTimed};
use crate::vectors::FeatureVector;
use crate::{
    FimlError, Float, HeapRingBuffer, IndicatorKind, InvalidArgumentError, Result, Symbol,
    WarmupPolicy,
};

pub(crate) struct ObvTimedFeature<F: Float> {
    symbol: Symbol,
    obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, MAX_OUTPUTS_PER_INDICATOR>,
}

impl<F: Float> ObvTimedFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self { symbol, obv }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output_span: OutputSpan,
        output: &mut O,
    ) {
        if let Event::Trade(trade) = event
            && trade.symbol == self.symbol
        {
            self.obv
                .update_inner(trade.price, trade.volume, trade.timestamp);
        } else if !self.obv.observe(event.timestamp()) {
            return;
        }

        write_outputs(output_span, output, |index| self.obv.window_value(index));
    }
}

pub(crate) fn build_timed<F: Float>(
    symbol: Symbol,
    aggregation: Duration,
    periods: &[usize],
    max_period: usize,
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation<F>> {
    let capacity = max_period.checked_add(1).ok_or(FimlError::InvalidArgument(
        InvalidArgumentError::TimedPeriodTooLarge {
            indicator: IndicatorKind::ObvTimed,
        },
    ))?;
    let mut obv =
        OnBalanceVolumeTimed::<
            HeapRingBuffer<ObvBucket<F>>,
            F,
            MAX_OUTPUTS_PER_INDICATOR,
        >::new_heap(aggregation, capacity, warmup_policy)?;
    for &period in periods {
        obv.add_window_with_periods(period)?;
    }
    Ok(FeatureDerivation::ObvTimed(ObvTimedFeature::new(
        symbol, obv,
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
    fn obv_timed_ingests_matching_trades_and_observes_other_events() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut obv: OnBalanceVolumeTimed<
            HeapRingBuffer<ObvBucket<f64>>,
            f64,
            MAX_OUTPUTS_PER_INDICATOR,
        > = OnBalanceVolumeTimed::new_heap(
            Duration::from_millis(1_000),
            3,
            WarmupPolicy::FirstValue,
        )
        .unwrap();
        obv.add_window_with_periods(2).unwrap();

        let mut feat = ObvTimedFeature::new(aapl, obv);
        let output_span = OutputSpan { start: 0, count: 1 };
        feat.update(
            &Event::trade(aapl, 100.0, 10.0, 0, None),
            output_span,
            &mut fv,
        );
        feat.update(
            &Event::trade(aapl, 101.0, 7.0, 1_000, None),
            output_span,
            &mut fv,
        );
        feat.update(
            &Event::trade(aapl, 99.0, 2.0, 2_000, None),
            output_span,
            &mut fv,
        );
        feat.update(&Event::price(aapl, 200.0, 3_000), output_span, &mut fv);
        feat.update(
            &Event::trade(googl, 110.0, 99.0, 3_000, None),
            output_span,
            &mut fv,
        );
        feat.update(&Event::time(3_000), output_span, &mut fv);

        assert!(approx_eq(fv.values()[0], -2.0));
    }
}

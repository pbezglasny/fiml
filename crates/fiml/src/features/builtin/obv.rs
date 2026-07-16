use std::time::Duration;

use crate::features::BuiltinFeature;
use crate::features::builtin::write_outputs;
use crate::features::compiler::OutputSpan;
use crate::features::definition::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::event::Event;
use crate::indicators::{ObvBucket, OnBalanceVolumeTimed};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, HeapRingBuffer, Result, Symbol};

pub struct ObvTimedFeature<F: Float> {
    symbol: Symbol,
    obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, MAX_OUTPUTS_PER_INDICATOR>,
    output_span: OutputSpan,
}

impl<F: Float> ObvTimedFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, MAX_OUTPUTS_PER_INDICATOR>,
        output_span: OutputSpan,
    ) -> Self {
        Self {
            symbol,
            obv,
            output_span,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Event::Trade(trade) = event
            && trade.symbol == self.symbol
        {
            self.obv
                .update_inner(trade.price, trade.volume, trade.timestamp);
            write_outputs(self.output_span, output, |index| {
                self.obv.window_value(index)
            });
        }
    }
}

pub(crate) fn build_timed<F: Float>(
    symbol: Symbol,
    aggregation: Duration,
    periods: &[usize],
    max_period: usize,
    output_span: OutputSpan,
) -> Result<BuiltinFeature<F>> {
    debug_assert_eq!(periods.len(), output_span.count);
    let capacity = max_period
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("OBV timed period is too large".to_string()))?;
    let mut obv =
        OnBalanceVolumeTimed::<
            HeapRingBuffer<ObvBucket<F>>,
            F,
            MAX_OUTPUTS_PER_INDICATOR,
        >::new_heap(aggregation, capacity)?;
    for &period in periods {
        obv.add_window_with_periods(period)?;
    }
    Ok(BuiltinFeature::ObvTimed(ObvTimedFeature::new(
        symbol,
        obv,
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
    fn obv_timed_reacts_to_trade_events() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut obv: OnBalanceVolumeTimed<
            HeapRingBuffer<ObvBucket<f64>>,
            f64,
            MAX_OUTPUTS_PER_INDICATOR,
        > = OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();

        let mut feat = ObvTimedFeature::new(aapl, obv, OutputSpan { start: 0, count: 1 });
        feat.update(&Event::trade(aapl, 100.0, 10.0, 0), &mut fv);
        feat.update(&Event::trade(aapl, 101.0, 7.0, 1_000), &mut fv);
        feat.update(&Event::trade(aapl, 99.0, 2.0, 2_000), &mut fv);
        feat.update(&Event::price(aapl, 200.0, 3_000), &mut fv);
        feat.update(&Event::trade(googl, 110.0, 99.0, 3_000), &mut fv);
        feat.update(&Event::time(3_000), &mut fv);

        assert!(approx_eq(fv.values()[0], 5.0));
    }
}

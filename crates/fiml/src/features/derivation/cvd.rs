use crate::event::Event;
use crate::features::compiler::OutputSpan;
use crate::features::definition::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::CumulativeVolumeDelta;
use crate::vectors::FeatureVector;
use crate::{Float, HeapRingBuffer, Result, Symbol, WarmupPolicy};

pub(crate) struct CvdFeature<F: Float> {
    symbol: Symbol,
    cvd: CumulativeVolumeDelta<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
}

impl<F: Float> CvdFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        cvd: CumulativeVolumeDelta<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self { symbol, cvd }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output_span: OutputSpan,
        output: &mut O,
    ) {
        if let Event::Trade(trade) = event
            && trade.symbol == self.symbol
            && let Some(side) = trade.side
        {
            self.cvd.update_inner(trade.volume, side);
            write_outputs(output_span, output, |index| self.cvd.value_at(index));
        }
    }
}

pub(crate) fn build<F: Float>(
    symbol: Symbol,
    windows: &[usize],
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation<F>> {
    let max_window = windows.iter().copied().max().unwrap_or(0);
    let mut cvd =
        CumulativeVolumeDelta::<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>::new_heap(
            max_window,
            warmup_policy,
        );
    for &window in windows {
        cvd.add_window(window)?;
    }
    Ok(FeatureDerivation::Cvd(CvdFeature::new(symbol, cvd)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::TradeSide;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    #[test]
    fn grouped_cvd_uses_trade_side_and_ignores_unclassified_trades() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut feature = match build::<f64>(aapl, &[1, 2], WarmupPolicy::FirstValue).unwrap() {
            FeatureDerivation::Cvd(feature) => feature,
            _ => unreachable!(),
        };
        let mut output = ArrayFeatureVector::<f64, 2>::new();
        let output_span = OutputSpan { start: 0, count: 2 };

        feature.update(
            &Event::trade(aapl, 100.0, 10.0, 0, Some(TradeSide::AgressorBuy)),
            output_span,
            &mut output,
        );
        feature.update(
            &Event::trade(aapl, 99.0, 3.0, 1, Some(TradeSide::AgressorSell)),
            output_span,
            &mut output,
        );
        feature.update(
            &Event::trade(aapl, 101.0, 50.0, 2, None),
            output_span,
            &mut output,
        );
        feature.update(
            &Event::trade(googl, 200.0, 80.0, 3, Some(TradeSide::AgressorBuy)),
            output_span,
            &mut output,
        );
        feature.update(
            &Event::trade(aapl, 102.0, 2.0, 4, Some(TradeSide::AgressorBuy)),
            output_span,
            &mut output,
        );

        assert_eq!(output.values(), [2.0, -1.0]);
    }
}

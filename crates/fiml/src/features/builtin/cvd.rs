use crate::features::BuiltinFeature;
use crate::features::builtin::write_outputs;
use crate::features::compiler::OutputSpan;
use crate::features::definition::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::event::Event;
use crate::indicators::CumulativeVolumeDelta;
use crate::vectors::FeatureVector;
use crate::{Float, HeapRingBuffer, Result, Symbol};

pub struct CvdFeature<F: Float> {
    symbol: Symbol,
    cvd: CumulativeVolumeDelta<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
    output_span: OutputSpan,
}

impl<F: Float> CvdFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        cvd: CumulativeVolumeDelta<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>,
        output_span: OutputSpan,
    ) -> Self {
        Self {
            symbol,
            cvd,
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
            && let Some(side) = trade.side
        {
            self.cvd.update_inner(trade.volume, side);
            write_outputs(self.output_span, output, |index| self.cvd.value_at(index));
        }
    }
}

pub(crate) fn build<F: Float>(
    symbol: Symbol,
    windows: &[usize],
    output_span: OutputSpan,
) -> Result<BuiltinFeature<F>> {
    debug_assert_eq!(windows.len(), output_span.count);
    let max_window = windows.iter().copied().max().unwrap_or(0);
    let mut cvd =
        CumulativeVolumeDelta::<HeapRingBuffer<F>, F, MAX_OUTPUTS_PER_INDICATOR>::new_heap(
            max_window,
        );
    for &window in windows {
        cvd.add_window(window)?;
    }
    Ok(BuiltinFeature::Cvd(CvdFeature::new(
        symbol,
        cvd,
        output_span,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::TradeSide;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    #[test]
    fn grouped_cvd_uses_trade_side_and_ignores_unclassified_trades() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut feature =
            match build::<f64>(aapl, &[1, 2], OutputSpan { start: 0, count: 2 }).unwrap() {
                BuiltinFeature::Cvd(feature) => feature,
                _ => unreachable!(),
            };
        let mut output = ArrayFeatureVector::<f64, 2>::new();

        feature.update(
            &Event::trade(aapl, 100.0, 10.0, 0, Some(TradeSide::AgressorBuy)),
            &mut output,
        );
        feature.update(
            &Event::trade(aapl, 99.0, 3.0, 1, Some(TradeSide::AgressorSell)),
            &mut output,
        );
        feature.update(&Event::trade(aapl, 101.0, 50.0, 2, None), &mut output);
        feature.update(
            &Event::trade(googl, 200.0, 80.0, 3, Some(TradeSide::AgressorBuy)),
            &mut output,
        );
        feature.update(
            &Event::trade(aapl, 102.0, 2.0, 4, Some(TradeSide::AgressorBuy)),
            &mut output,
        );

        assert_eq!(output.values(), [2.0, -1.0]);
    }
}

use std::time::Duration;

use crate::features::BuiltinFeature;
use crate::features::compiler::OutputSpan;
use crate::features::event::Event;
use crate::indicators::{CountBucket, TradeCountTimed};
use crate::vectors::FeatureVector;
use crate::{Float, HeapRingBuffer, Result, Symbol};

/// Rolling count of trades over a time window, wired to one output cell. Reacts
/// to [`Trade`](EventKind::Trade) events for its symbol.
pub struct TradeCountTimedFeature<F: Float> {
    symbol: Symbol,
    counter: TradeCountTimed<HeapRingBuffer<CountBucket>, F>,
    output_span: OutputSpan,
}

impl<F: Float> TradeCountTimedFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        counter: TradeCountTimed<HeapRingBuffer<CountBucket>, F>,
        output_span: OutputSpan,
    ) -> Self {
        debug_assert_eq!(output_span.count, 1);
        Self {
            symbol,
            counter,
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
            self.counter.update_inner(trade.timestamp);
            output.set_value_at(self.output_span.start, self.counter.window_value());
        }
    }
}

pub(crate) fn build<F: Float>(
    symbol: Symbol,
    aggregation: Duration,
    window: Duration,
    output_span: OutputSpan,
) -> Result<BuiltinFeature<F>> {
    let counter = TradeCountTimed::<HeapRingBuffer<CountBucket>, F>::new_heap(aggregation, window)?;
    Ok(BuiltinFeature::TradeCountTimed(
        TradeCountTimedFeature::new(symbol, counter, output_span),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn counts_only_trades_for_its_symbol() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let counter = TradeCountTimed::<HeapRingBuffer<CountBucket>, f64>::new_heap(
            Duration::from_millis(1_000),
            Duration::from_millis(2_000),
        )
        .unwrap();
        let mut feat =
            TradeCountTimedFeature::new(aapl, counter, OutputSpan { start: 0, count: 1 });

        feat.update(&Event::trade(aapl, 100.0, 1.0, 0), &mut fv);
        feat.update(&Event::trade(aapl, 101.0, 1.0, 100), &mut fv);
        feat.update(&Event::trade(googl, 50.0, 1.0, 200), &mut fv); // other symbol
        feat.update(&Event::price(aapl, 102.0, 300), &mut fv); // other kind

        assert!(approx_eq(fv.values()[0], 2.0));
    }
}

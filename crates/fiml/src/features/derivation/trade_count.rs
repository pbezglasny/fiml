use std::time::Duration;

use crate::event::Event;
use crate::features::compiler::OutputRange;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::{CountBucket, TradeCountTimed};
use crate::vectors::FeatureVector;
use crate::{HeapRingBuffer, Result, Symbol, WarmupPolicy};

/// Rolling count of trades over a time window, wired to one output cell. Reacts
/// to [`Trade`](crate::event::EventKind::Trade) events for its symbol.
pub(crate) struct TradeCountTimedFeature {
    symbol: Symbol,
    counter: TradeCountTimed<HeapRingBuffer<CountBucket>>,
}

impl TradeCountTimedFeature {
    pub(crate) fn new(
        symbol: Symbol,
        counter: TradeCountTimed<HeapRingBuffer<CountBucket>>,
    ) -> Self {
        Self { symbol, counter }
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
            self.counter.update(trade.timestamp);
        } else if !self.counter.observe(event.timestamp()) {
            return;
        }

        write_outputs(output_range, output, |_| self.counter.window_value());
    }
}

pub(crate) fn build(
    symbol: Symbol,
    aggregation: Duration,
    window: Duration,
    warmup_policy: WarmupPolicy,
) -> Result<FeatureDerivation> {
    let counter = TradeCountTimed::<HeapRingBuffer<CountBucket>>::new_heap(
        aggregation,
        window,
        warmup_policy,
    )?;
    Ok(FeatureDerivation::TradeCountTimed(
        TradeCountTimedFeature::new(symbol, counter),
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
        let aapl = symbols::intern("AAPL").unwrap();
        let googl = symbols::intern("GOOGL").unwrap();
        let mut fv: ArrayFeatureVector<1> = ArrayFeatureVector::new();
        let counter = TradeCountTimed::<HeapRingBuffer<CountBucket>>::new_heap(
            Duration::from_millis(1_000),
            Duration::from_millis(2_000),
            WarmupPolicy::FirstValue,
        )
        .unwrap();
        let mut feat = TradeCountTimedFeature::new(aapl, counter);
        let output_range = OutputRange { start: 0, count: 1 };

        feat.update(
            &Event::trade(aapl, 100.0, 1.0, 0, None),
            output_range,
            &mut fv,
        );
        feat.update(
            &Event::trade(aapl, 101.0, 1.0, 100, None),
            output_range,
            &mut fv,
        );
        feat.update(
            &Event::trade(googl, 50.0, 1.0, 200, None),
            output_range,
            &mut fv,
        ); // other symbol
        feat.update(&Event::price(aapl, 102.0, 300), output_range, &mut fv); // other kind

        assert!(approx_eq(fv.values()[0], 2.0));
    }
}

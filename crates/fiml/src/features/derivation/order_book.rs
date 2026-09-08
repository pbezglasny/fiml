//! Writes existing order-book calculations directly into compiled output spans.

use rust_decimal::prelude::ToPrimitive;

use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputSpan;
use crate::features::derivation::write_outputs;
use crate::{FeatureVector, Symbol, order_book::OrderBook};

/// Calculation selected at compilation, with fixed storage for grouped depths.
pub(crate) enum OrderBookIndicator {
    MidPrice,
    Spread,
    SpreadBps,
    WeightedMidPrice,
    Microprice,
    Imbalance([usize; MAX_OUTPUTS_PER_INDICATOR]),
}

/// Reads one configured book without owning or duplicating its market state.
pub(crate) struct OrderBookFeature {
    pub(crate) symbol: Symbol,
    pub(crate) indicator: OrderBookIndicator,
}

impl OrderBookFeature {
    pub(crate) fn update<O: FeatureVector>(
        &self,
        book: &OrderBook,
        span: OutputSpan,
        output: &mut O,
    ) {
        write_outputs(span, output, |index| {
            let value = match &self.indicator {
                OrderBookIndicator::MidPrice => book.mid_price(),
                OrderBookIndicator::Spread => book.spread(),
                OrderBookIndicator::SpreadBps => book.spread_bps(),
                OrderBookIndicator::WeightedMidPrice => book.weighted_mid_price(),
                OrderBookIndicator::Microprice => book.microprice(),
                // ponytail: overlapping depths repeat scans; share traversal if profiling warrants it.
                OrderBookIndicator::Imbalance(depths) => book.imbalance(depths[index]),
            };
            value.and_then(|value| value.to_f64())
        });
    }
}

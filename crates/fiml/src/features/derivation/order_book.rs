//! Writes existing order-book calculations directly into compiled output ranges.

use rust_decimal::{Decimal, prelude::ToPrimitive};

use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputRange;
use crate::features::derivation::write_outputs;
use crate::{
    FeatureVector, Symbol,
    order_book::{OrderBook, Side},
};

/// Calculation selected at compilation, with fixed storage for grouped depths.
pub(crate) enum OrderBookIndicator {
    LevelSize {
        side: Side,
        price: Decimal,
    },
    NthPrice {
        side: Side,
        n_levels: usize,
    },
    NthSize {
        side: Side,
        n_levels: usize,
    },
    DepthUntilPrice {
        side: Side,
        price: Decimal,
    },
    DepthUntilSizePriceFrom {
        side: Side,
        size: Decimal,
    },
    DepthUntilSizePriceTo {
        side: Side,
        size: Decimal,
    },
    DepthUntilSizeTotalSize {
        side: Side,
        size: Decimal,
    },
    VolumeBetweenPrices {
        side: Side,
        from_price: Decimal,
        to_price: Decimal,
    },
    BestBidPrice,
    BestBidSize,
    BestAskPrice,
    BestAskSize,
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
        output_range: OutputRange,
        output: &mut O,
    ) {
        // ponytail: independent depth queries repeat scans; share traversal if profiling warrants it.
        write_outputs(output_range, output, |index| {
            let value = match &self.indicator {
                OrderBookIndicator::LevelSize { side, price } => book.level(*side, *price),
                OrderBookIndicator::NthPrice { side, n_levels } => book
                    .top_n(*side, *n_levels)
                    .nth(n_levels - 1)
                    .map(|level| level.price),
                OrderBookIndicator::NthSize { side, n_levels } => book
                    .top_n(*side, *n_levels)
                    .nth(n_levels - 1)
                    .map(|level| level.size),
                OrderBookIndicator::DepthUntilPrice { side, price } => {
                    Some(book.depth_until_price(*side, *price))
                }
                OrderBookIndicator::DepthUntilSizePriceFrom { side, size } => book
                    .depth_until_total_size(*side, *size)
                    .map(|depth| depth.price_from),
                OrderBookIndicator::DepthUntilSizePriceTo { side, size } => book
                    .depth_until_total_size(*side, *size)
                    .map(|depth| depth.price_to),
                OrderBookIndicator::DepthUntilSizeTotalSize { side, size } => book
                    .depth_until_total_size(*side, *size)
                    .map(|depth| depth.total_size),
                OrderBookIndicator::VolumeBetweenPrices {
                    side,
                    from_price,
                    to_price,
                } => book
                    .volume_between_prices(*side, *from_price, *to_price)
                    .ok(),
                OrderBookIndicator::BestBidPrice => book.best_bid().map(|level| level.price),
                OrderBookIndicator::BestBidSize => book.best_bid().map(|level| level.size),
                OrderBookIndicator::BestAskPrice => book.best_ask().map(|level| level.price),
                OrderBookIndicator::BestAskSize => book.best_ask().map(|level| level.size),
                OrderBookIndicator::MidPrice => book.mid_price(),
                OrderBookIndicator::Spread => book.spread(),
                OrderBookIndicator::SpreadBps => book.spread_bps(),
                OrderBookIndicator::WeightedMidPrice => book.weighted_mid_price(),
                OrderBookIndicator::Microprice => book.microprice(),
                OrderBookIndicator::Imbalance(depths) => book.imbalance(depths[index]),
            };
            value.and_then(|value| value.to_f64())
        });
    }
}

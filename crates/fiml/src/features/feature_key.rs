use std::time::Duration;

use rust_decimal::Decimal;

use crate::order_book::Side;

use crate::{Symbol, WarmupPolicy};

use super::feature_source::FeatureSource;

/// Structured identity of one scalar feature-vector output.
///
/// Equal keys describe the same feature. Indicators with multiple windows
/// therefore have one key per window rather than one key for the complete
/// runtime indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureKey {
    /// Size at an exact price; a missing level produces `NaN`.
    OrderBookLevelSize {
        symbol: Symbol,
        side: Side,
        price: Decimal,
    },
    /// Price at the one-based depth `n_levels`; missing levels produce `NaN`.
    OrderBookNthPrice {
        symbol: Symbol,
        side: Side,
        n_levels: usize,
    },
    /// Size at the one-based depth `n_levels`; missing levels produce `NaN`.
    OrderBookNthSize {
        symbol: Symbol,
        side: Side,
        n_levels: usize,
    },
    /// Total size from the best quote through an inclusive price threshold; empty depth is zero.
    OrderBookDepthUntilPrice {
        symbol: Symbol,
        side: Side,
        price: Decimal,
    },
    /// First price needed to reach a positive size; insufficient depth produces `NaN`.
    OrderBookDepthUntilSizePriceFrom {
        symbol: Symbol,
        side: Side,
        size: Decimal,
    },
    /// Last price needed to reach a positive size; insufficient depth produces `NaN`.
    OrderBookDepthUntilSizePriceTo {
        symbol: Symbol,
        side: Side,
        size: Decimal,
    },
    /// Whole-level cumulative size needed to reach a positive target; insufficient depth produces `NaN`.
    OrderBookDepthUntilSizeTotalSize {
        symbol: Symbol,
        side: Side,
        size: Decimal,
    },
    /// Total size in `[from_price, to_price)`; requires nonnegative prices and `from_price < to_price`.
    OrderBookVolumeBetweenPrices {
        symbol: Symbol,
        side: Side,
        from_price: Decimal,
        to_price: Decimal,
    },
    /// Price of the best bid; unavailable sides produce `NaN`.
    OrderBookBestBidPrice { symbol: Symbol },
    /// Size of the best bid; unavailable sides produce `NaN`.
    OrderBookBestBidSize { symbol: Symbol },
    /// Price of the best ask; unavailable sides produce `NaN`.
    OrderBookBestAskPrice { symbol: Symbol },
    /// Size of the best ask; unavailable sides produce `NaN`.
    OrderBookBestAskSize { symbol: Symbol },
    /// Midpoint of the best bid and ask in a configured order book.
    OrderBookMidPrice { symbol: Symbol },
    /// Absolute spread between the best bid and ask.
    OrderBookSpread { symbol: Symbol },
    /// Absolute spread divided by mid-price, in basis points.
    OrderBookSpreadBps { symbol: Symbol },
    /// Best bid and ask prices weighted by their own sizes.
    OrderBookWeightedMidPrice { symbol: Symbol },
    /// Best bid and ask prices weighted by the opposite side's size.
    OrderBookMicroprice { symbol: Symbol },
    /// Bid-minus-ask size divided by total size over up to `n_levels` per side.
    /// Depth must be positive; available levels are used if the book is shorter.
    OrderBookImbalance { symbol: Symbol, n_levels: usize },
    Sma {
        symbol: Symbol,
        source: FeatureSource,
        window: usize,
        warmup_policy: WarmupPolicy,
    },
    Ema {
        symbol: Symbol,
        source: FeatureSource,
        window: usize,
        warmup_policy: WarmupPolicy,
    },
    Cvd {
        symbol: Symbol,
        source: FeatureSource,
        window: usize,
        warmup_policy: WarmupPolicy,
    },
    SmaTimed {
        symbol: Symbol,
        source: FeatureSource,
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    },
    ObvTimed {
        symbol: Symbol,
        source: FeatureSource,
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    },
    /// Cumulative volume-price trend over trades for one symbol.
    Vpt {
        symbol: Symbol,
        source: FeatureSource,
    },
    TradeCountTimed {
        symbol: Symbol,
        source: FeatureSource,
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    },
    DayOfWeek {
        symbol: Symbol,
        source: FeatureSource,
    },
    TimeSinceFirstEventOfDay {
        symbol: Symbol,
        source: FeatureSource,
        utc_offset_millis: i64,
    },
}

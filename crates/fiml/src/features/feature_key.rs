//! Defines the structural identity of every supported scalar feature output.
//!
use std::time::Duration;

use rust_decimal::Decimal;

use crate::order_book::Side;

use crate::{ReturnKind, Symbol, WarmupPolicy};

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
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Nonnegative price to query.
        price: Decimal,
    },
    /// Price at the one-based depth `n_levels`; missing levels produce `NaN`.
    OrderBookNthPrice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Positive, one-based depth or number of levels to include.
        n_levels: usize,
    },
    /// Size at the one-based depth `n_levels`; missing levels produce `NaN`.
    OrderBookNthSize {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Positive, one-based depth or number of levels to include.
        n_levels: usize,
    },
    /// Total size from the best quote through an inclusive price threshold; empty depth is zero.
    OrderBookDepthUntilPrice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Nonnegative price to query.
        price: Decimal,
    },
    /// First price needed to reach a positive size; insufficient depth produces `NaN`.
    OrderBookDepthUntilSizePriceFrom {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Positive cumulative size to reach.
        size: Decimal,
    },
    /// Last price needed to reach a positive size; insufficient depth produces `NaN`.
    OrderBookDepthUntilSizePriceTo {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Positive cumulative size to reach.
        size: Decimal,
    },
    /// Whole-level cumulative size needed to reach a positive target; insufficient depth produces `NaN`.
    OrderBookDepthUntilSizeTotalSize {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Positive cumulative size to reach.
        size: Decimal,
    },
    /// Total size in `[from_price, to_price)`; requires nonnegative prices and `from_price < to_price`.
    OrderBookVolumeBetweenPrices {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Bid or ask side to query.
        side: Side,
        /// Inclusive, nonnegative lower price bound.
        from_price: Decimal,
        /// Exclusive upper price bound; must exceed `from_price`.
        to_price: Decimal,
    },
    /// Price of the best bid; unavailable sides produce `NaN`.
    OrderBookBestBidPrice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Size of the best bid; unavailable sides produce `NaN`.
    OrderBookBestBidSize {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Price of the best ask; unavailable sides produce `NaN`.
    OrderBookBestAskPrice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Size of the best ask; unavailable sides produce `NaN`.
    OrderBookBestAskSize {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Midpoint of the best bid and ask in a configured order book.
    OrderBookMidPrice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Absolute spread between the best bid and ask.
    OrderBookSpread {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Absolute spread divided by mid-price, in basis points.
    OrderBookSpreadBps {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Best bid and ask prices weighted by their own sizes.
    OrderBookWeightedMidPrice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Best bid and ask prices weighted by the opposite side's size.
    OrderBookMicroprice {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
    },
    /// Bid-minus-ask size divided by total size over up to `n_levels` per side.
    /// Depth must be positive; available levels are used if the book is shorter.
    OrderBookImbalance {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Positive, one-based depth or number of levels to include.
        n_levels: usize,
    },
    /// Latest selected scalar field, retained until the next matching event.
    Field {
        /// Market symbol supplying the observation.
        symbol: Symbol,
        /// Scalar event field to copy.
        field: crate::EventField,
    },
    /// Rolling aggressor buy volume minus sell volume.
    Cvd {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Event(EventKind::Trade)`.
        source: FeatureSource,
        /// Positive number of samples in the rolling window.
        window: usize,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Simple or logarithmic return from the current value to a lagged sample.
    Return {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Field` selecting a scalar event field.
        source: FeatureSource,
        /// Simple fractional change or natural-log price ratio.
        kind: ReturnKind,
        /// Positive sample distance to the earlier value.
        lag: usize,
    },
    /// Population volatility of simple returns over a sample window.
    Volatility {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Field` selecting a scalar event field.
        source: FeatureSource,
        /// Positive number of simple returns; requires one additional price sample.
        window: usize,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Simple moving average over time buckets.
    SmaTimed {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Field` selecting a scalar event field.
        source: FeatureSource,
        /// Positive bucket duration in whole milliseconds, representable as `i64` milliseconds.
        aggregation: Duration,
        /// Rolling duration: a positive multiple of `aggregation`, in whole `i64` milliseconds.
        window: Duration,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Population volatility of simple returns over a timed window.
    VolatilityTimed {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Field` selecting a scalar event field.
        source: FeatureSource,
        /// Positive bucket duration in whole milliseconds, representable as `i64` milliseconds.
        aggregation: Duration,
        /// Rolling duration: a positive multiple of `aggregation`, in whole `i64` milliseconds.
        window: Duration,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// On-balance volume over time buckets.
    ObvTimed {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Event(EventKind::Trade)`.
        source: FeatureSource,
        /// Positive bucket duration in whole milliseconds, representable as `i64` milliseconds.
        aggregation: Duration,
        /// Rolling duration: a positive multiple of `aggregation`, in whole `i64` milliseconds.
        window: Duration,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Cumulative volume-price trend over trades for one symbol.
    Vpt {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Event(EventKind::Trade)`.
        source: FeatureSource,
    },
    /// Trade count over a rolling timed window.
    TradeCountTimed {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Event(EventKind::Trade)`.
        source: FeatureSource,
        /// Positive bucket duration in whole milliseconds, representable as `i64` milliseconds.
        aggregation: Duration,
        /// Rolling duration: a positive multiple of `aggregation`, in whole `i64` milliseconds.
        window: Duration,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Sum of trade volume over a time-bucketed rolling window.
    TradeVolumeTimed {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Event(EventKind::Trade)`.
        source: FeatureSource,
        /// Positive bucket duration in whole milliseconds, representable as `i64` milliseconds.
        aggregation: Duration,
        /// Rolling duration: a positive multiple of `aggregation`, in whole `i64` milliseconds.
        window: Duration,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Volume-weighted average trade price over a time-bucketed rolling window.
    VwapTimed {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Must be `FeatureSource::Event(EventKind::Trade)`.
        source: FeatureSource,
        /// Positive bucket duration in whole milliseconds, representable as `i64` milliseconds.
        aggregation: Duration,
        /// Rolling duration: a positive multiple of `aggregation`, in whole `i64` milliseconds.
        window: Duration,
        /// Controls when output becomes available while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// UTC weekday from the event timestamp: `0 = Sunday` through `6 = Saturday`.
    DayOfWeek {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Event kinds or field-bearing events whose timestamps drive this feature.
        source: FeatureSource,
    },
    /// Milliseconds since the first observed event of the fixed-offset local day.
    TimeSinceFirstEventOfDay {
        /// Market symbol whose events or order book supply this output.
        symbol: Symbol,
        /// Event kinds or field-bearing events whose timestamps drive this feature.
        source: FeatureSource,
        /// Local-day UTC offset in whole minutes, within -14 to +14 hours inclusive.
        utc_offset_millis: i64,
    },
}

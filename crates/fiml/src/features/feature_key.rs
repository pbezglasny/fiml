use std::time::Duration;

use crate::{Symbol, WarmupPolicy};

use super::feature_source::FeatureSource;

/// Structured identity of one scalar feature-vector output.
///
/// Equal keys describe the same feature. Indicators with multiple windows
/// therefore have one key per window rather than one key for the complete
/// runtime indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureKey {
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

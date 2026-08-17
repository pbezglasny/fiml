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

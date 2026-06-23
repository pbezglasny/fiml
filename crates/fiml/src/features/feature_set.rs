use crate::features::event::{EventKind, FeatureRoute};
use std::time::Duration;

/// Time unit for windows of time-based features. Used in the [`IndicatorSpec::Sma`] variant etc
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimeUnit {
    Sec,
    Min,
    Hour,
}

/// Structured description of a library-provided feature, consumed by
/// [`IndicatorFeatureVector::from_builtin_specs`](crate::features::IndicatorFeatureVector::from_builtin_specs).
///
/// The string parser that turns feature names into specs will be added later;
/// for now specs are constructed directly.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndicatorSpec {
    /// Simple moving average over `period`
    Sma { period: usize },
    /// Exponential moving average over `period`
    Ema { period: usize },
    /// Time-bucketed simple moving average over `window`, using `aggregation`
    /// as the bucket size. Price event timestamps must be milliseconds.
    SmaTimed {
        aggregation: Duration,
        window: Duration,
    },
    /// Time-bucketed on-balance volume over `window`, using `aggregation` as
    /// the bucket size. Trade event timestamps must be milliseconds.
    ObvTimed {
        aggregation: Duration,
        window: Duration,
    },
    /// Rolling count of trades over `window`, bucketed by `aggregation`. Trade
    /// event timestamps must be milliseconds.
    TradeCountTimed {
        aggregation: Duration,
        window: Duration,
    },
    /// Day-of-week clock feature (`0 = Sunday ..= 6 = Saturday`). Updates on
    /// every event from its timestamp.
    DayOfWeek,
    /// Milliseconds since the trading session opened (the first event after a day
    /// boundary). The day boundary is taken in a fixed UTC offset
    /// (`utc_offset_millis`, `0` = UTC). Updates on every event.
    TimeSinceSessionOpen { utc_offset_millis: i64 },
}

impl IndicatorSpec {
    /// Dispatch route the resulting feature subscribes to. Must agree with the
    /// matching [`BuiltinFeature`](crate::features::BuiltinFeature) variant.
    /// Clock features (`DayOfWeek`) subscribe to every event so they refresh on
    /// every dispatch.
    pub fn route(&self) -> FeatureRoute {
        match self {
            IndicatorSpec::Sma { .. } => FeatureRoute::Kind(EventKind::Price),
            IndicatorSpec::Ema { .. } => FeatureRoute::Kind(EventKind::Price),
            IndicatorSpec::SmaTimed { .. } => FeatureRoute::Kind(EventKind::Price),
            IndicatorSpec::ObvTimed { .. } => FeatureRoute::Kind(EventKind::Trade),
            IndicatorSpec::TradeCountTimed { .. } => FeatureRoute::Kind(EventKind::Trade),
            IndicatorSpec::DayOfWeek => FeatureRoute::Every,
            IndicatorSpec::TimeSinceSessionOpen { .. } => FeatureRoute::Every,
        }
    }
}

/// One output column of an engine: the cell `name`, the `symbol` it tracks
/// (stored as the symbol's string name so it survives serialization across
/// processes), and the [`IndicatorSpec`] describing the feature.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureDef {
    pub name: String,
    pub symbol: String,
    pub indicator: IndicatorSpec,
}

/// Declarative, serializable description of a whole engine: the ordered list of
/// features it produces.
///
/// This is the parity contract between batch (e.g. Python training) and live
/// (Rust serving): build both engines from the same `FeatureSet` and feed the
/// same events in the same order to get identical output. With the `serde`
/// feature it round-trips through JSON so the exact configuration can be saved
/// next to a trained model and reloaded. Build a runnable engine with
/// [`FeatureExtractor::from_feature_set`](crate::features::FeatureExtractor::from_feature_set).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureSet {
    pub features: Vec<FeatureDef>,
}

impl FeatureSet {
    /// Create a spec from an explicit list of features.
    pub fn new(features: Vec<FeatureDef>) -> Self {
        Self { features }
    }
}

use crate::features::event::EventKind;
use std::time::Duration;

/// Time unit for windows of time-based features. Used in the [`BuiltinSpec::Sma`] variant etc
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
pub enum BuiltinSpec {
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
    /// Day-of-week non-price feature.
    DayOfWeek,
}

impl BuiltinSpec {
    /// Event kind the resulting feature subscribes to. Must agree with the
    /// matching [`BuiltinFeature`](crate::features::BuiltinFeature) variant.
    pub fn event_kind(&self) -> EventKind {
        match self {
            BuiltinSpec::Sma { .. } => EventKind::Price,
            BuiltinSpec::Ema { .. } => EventKind::Price,
            BuiltinSpec::SmaTimed { .. } => EventKind::Price,
            BuiltinSpec::ObvTimed { .. } => EventKind::Trade,
            BuiltinSpec::DayOfWeek => EventKind::Time,
        }
    }
}

/// One output column of an engine: the cell `name`, the `symbol` it tracks
/// (stored as the symbol's string name so it survives serialization across
/// processes), and the [`BuiltinSpec`] describing the feature.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureSpec {
    pub name: String,
    pub symbol: String,
    pub spec: BuiltinSpec,
}

/// Declarative, serializable description of a whole engine: the ordered list of
/// features it produces.
///
/// This is the parity contract between batch (e.g. Python training) and live
/// (Rust serving): build both engines from the same `EngineSpec` and feed the
/// same events in the same order to get identical output. With the `serde`
/// feature it round-trips through JSON so the exact configuration can be saved
/// next to a trained model and reloaded. Build a runnable engine with
/// [`DynIndicatorEngine::from_spec`](crate::features::DynIndicatorEngine::from_spec).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineSpec {
    pub features: Vec<FeatureSpec>,
}

impl EngineSpec {
    /// Create a spec from an explicit list of features.
    pub fn new(features: Vec<FeatureSpec>) -> Self {
        Self { features }
    }
}

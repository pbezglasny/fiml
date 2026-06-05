use crate::features::event::EventKind;
use std::time::Duration;

/// Time unit for windows of time-based features. Used in the [`BuiltinSpec::Sma`] variant etc
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            BuiltinSpec::DayOfWeek => EventKind::Time,
        }
    }
}

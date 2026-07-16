use std::time::Duration;

use crate::features::event::{Event, EventKind, FeatureRoute};
use crate::{Float, Symbol};

/// Maximum number of adjacent outputs one runtime indicator may own.
pub const MAX_OUTPUTS_PER_INDICATOR: usize = 16;

/// Numeric event field consumed by a moving average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ValueSource {
    #[default]
    Price,
    Volume,
    TradePrice,
    TradeVolume,
}

impl ValueSource {
    pub(crate) fn route(self) -> FeatureRoute {
        FeatureRoute::Kind(match self {
            Self::Price => EventKind::Price,
            Self::Volume => EventKind::Volume,
            Self::TradePrice | Self::TradeVolume => EventKind::Trade,
        })
    }

    pub(crate) fn canonical_name(self) -> &'static str {
        match self {
            Self::Price => "price",
            Self::Volume => "volume",
            Self::TradePrice => "trade_price",
            Self::TradeVolume => "trade_volume",
        }
    }

    pub(crate) fn value<F: Float>(self, event: &Event<F>, symbol: Symbol) -> Option<F> {
        match (self, event) {
            (Self::Price, Event::Price(update)) if update.symbol == symbol => Some(update.value),
            (Self::Volume, Event::Volume(update)) if update.symbol == symbol => Some(update.value),
            (Self::TradePrice, Event::Trade(update)) if update.symbol == symbol => {
                Some(update.price)
            }
            (Self::TradeVolume, Event::Trade(update)) if update.symbol == symbol => {
                Some(update.volume)
            }
            _ => None,
        }
    }
}

/// Bucket aggregation and ordered rolling windows for a timed indicator.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeWindows {
    pub aggregation: Duration,
    pub windows: Vec<Duration>,
}

impl TimeWindows {
    pub fn new(aggregation: Duration, windows: Vec<Duration>) -> Self {
        Self {
            aggregation,
            windows,
        }
    }
}

/// Structured configuration for one runtime indicator instance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndicatorSpec {
    Sma {
        source: ValueSource,
        windows: Vec<usize>,
    },
    Ema {
        source: ValueSource,
        windows: Vec<usize>,
    },
    SmaTimed {
        source: ValueSource,
        time_windows: TimeWindows,
    },
    ObvTimed {
        time_windows: TimeWindows,
    },
    TradeCountTimed {
        aggregation: Duration,
        window: Duration,
    },
    DayOfWeek,
    TimeSinceFirstEventOfDay {
        utc_offset_millis: i64,
    },
}

impl IndicatorSpec {
    pub fn output_count(&self) -> usize {
        match self {
            Self::Sma { windows, .. } | Self::Ema { windows, .. } => windows.len(),
            Self::SmaTimed { time_windows, .. } | Self::ObvTimed { time_windows } => {
                time_windows.windows.len()
            }
            Self::TradeCountTimed { .. }
            | Self::DayOfWeek
            | Self::TimeSinceFirstEventOfDay { .. } => 1,
        }
    }

    pub(crate) fn route(&self) -> FeatureRoute {
        match self {
            Self::Sma { source, .. } | Self::Ema { source, .. } | Self::SmaTimed { source, .. } => {
                source.route()
            }
            Self::ObvTimed { .. } | Self::TradeCountTimed { .. } => {
                FeatureRoute::Kind(EventKind::Trade)
            }
            Self::DayOfWeek | Self::TimeSinceFirstEventOfDay { .. } => FeatureRoute::Every,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Sma { .. } => "SMA",
            Self::Ema { .. } => "EMA",
            Self::SmaTimed { .. } => "timed SMA",
            Self::ObvTimed { .. } => "timed OBV",
            Self::TradeCountTimed { .. } => "timed trade count",
            Self::DayOfWeek => "day of week",
            Self::TimeSinceFirstEventOfDay { .. } => "time since first event of day",
        }
    }

    pub(crate) fn is_global(&self) -> bool {
        matches!(
            self,
            Self::DayOfWeek | Self::TimeSinceFirstEventOfDay { .. }
        )
    }
}

/// One user-authored runtime indicator definition.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndicatorDef {
    /// Symbol name for market indicators. Global clock indicators use `None`.
    pub symbol: Option<String>,
    pub indicator: IndicatorSpec,
}

impl IndicatorDef {
    pub fn symbol(symbol: impl Into<String>, indicator: IndicatorSpec) -> Self {
        Self {
            symbol: Some(symbol.into()),
            indicator,
        }
    }

    pub fn global(indicator: IndicatorSpec) -> Self {
        Self {
            symbol: None,
            indicator,
        }
    }
}

/// Ordered, serializable definitions for a complete extractor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureSet {
    pub indicators: Vec<IndicatorDef>,
}

impl FeatureSet {
    pub fn new(indicators: Vec<IndicatorDef>) -> Self {
        Self { indicators }
    }

    pub fn indicator_count(&self) -> usize {
        self.indicators.len()
    }

    pub fn output_count(&self) -> usize {
        self.indicators
            .iter()
            .map(|definition| definition.indicator.output_count())
            .sum()
    }
}

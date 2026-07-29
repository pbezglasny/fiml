use std::cmp::Ordering;
use std::time::Duration;

use crate::features::event::{Event, EventKind, FeatureRoute};
use crate::{Float, Symbol, WarmupPolicy};

/// Maximum number of adjacent outputs one runtime indicator may own.
pub const MAX_OUTPUTS_PER_INDICATOR: usize = 16;

/// Numeric event field consumed by a moving average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
pub enum IndicatorSpec {
    Sma {
        source: ValueSource,
        windows: Vec<usize>,
        warmup_policy: WarmupPolicy,
    },
    Ema {
        source: ValueSource,
        windows: Vec<usize>,
        warmup_policy: WarmupPolicy,
    },
    Cvd {
        windows: Vec<usize>,
        warmup_policy: WarmupPolicy,
    },
    SmaTimed {
        source: ValueSource,
        time_windows: TimeWindows,
        warmup_policy: WarmupPolicy,
    },
    ObvTimed {
        time_windows: TimeWindows,
        warmup_policy: WarmupPolicy,
    },
    TradeCountTimed {
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    },
    DayOfWeek,
    TimeSinceFirstEventOfDay {
        utc_offset_millis: i64,
    },
}

impl IndicatorSpec {
    pub fn output_count(&self) -> usize {
        match self {
            Self::Sma { windows, .. } | Self::Ema { windows, .. } | Self::Cvd { windows, .. } => {
                windows.len()
            }
            Self::SmaTimed { time_windows, .. } | Self::ObvTimed { time_windows, .. } => {
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
            Self::Cvd { .. } | Self::ObvTimed { .. } | Self::TradeCountTimed { .. } => {
                FeatureRoute::Kind(EventKind::Trade)
            }
            Self::DayOfWeek | Self::TimeSinceFirstEventOfDay { .. } => FeatureRoute::Every,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Sma { .. } => "SMA",
            Self::Ema { .. } => "EMA",
            Self::Cvd { .. } => "CVD",
            Self::SmaTimed { .. } => "timed SMA",
            Self::ObvTimed { .. } => "timed OBV",
            Self::TradeCountTimed { .. } => "timed trade count",
            Self::DayOfWeek => "day of week",
            Self::TimeSinceFirstEventOfDay { .. } => "time since first event of day",
        }
    }

    pub(crate) fn canonical_name(&self) -> &'static str {
        match self {
            Self::Sma { .. } => "sma",
            Self::Ema { .. } => "ema",
            Self::Cvd { .. } => "cvd",
            Self::SmaTimed { .. } => "sma_timed",
            Self::ObvTimed { .. } => "obv_timed",
            Self::TradeCountTimed { .. } => "trade_count_timed",
            Self::DayOfWeek => "day_of_week",
            Self::TimeSinceFirstEventOfDay { .. } => "time_since_first_event_of_day",
        }
    }

    fn identity_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (
                Self::Sma { source: left, .. } | Self::Ema { source: left, .. },
                Self::Sma { source: right, .. } | Self::Ema { source: right, .. },
            ) => left.canonical_name().cmp(right.canonical_name()),
            (
                Self::SmaTimed {
                    source: left_source,
                    time_windows: left_windows,
                    ..
                },
                Self::SmaTimed {
                    source: right_source,
                    time_windows: right_windows,
                    ..
                },
            ) => left_source
                .canonical_name()
                .cmp(right_source.canonical_name())
                .then_with(|| left_windows.aggregation.cmp(&right_windows.aggregation)),
            (
                Self::ObvTimed {
                    time_windows: left, ..
                },
                Self::ObvTimed {
                    time_windows: right,
                    ..
                },
            ) => left.aggregation.cmp(&right.aggregation),
            (
                Self::TradeCountTimed {
                    aggregation: left, ..
                },
                Self::TradeCountTimed {
                    aggregation: right, ..
                },
            ) => left.cmp(right),
            (
                Self::TimeSinceFirstEventOfDay {
                    utc_offset_millis: left,
                },
                Self::TimeSinceFirstEventOfDay {
                    utc_offset_millis: right,
                },
            ) => left.cmp(right),
            _ => Ordering::Equal,
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
pub struct IndicatorDef {
    /// Symbol name for market indicators. Global clock indicators use `None`.
    pub symbol: Option<String>,
    pub indicator: IndicatorSpec,
}

impl IndicatorDef {
    pub fn symbol(symbol: impl Into<String>, indicator: IndicatorSpec) -> Self {
        let symbol = symbol.into();
        Self {
            symbol: Some(crate::symbols::normalize_name(&symbol).into_owned()),
            indicator,
        }
    }

    pub fn global(indicator: IndicatorSpec) -> Self {
        Self {
            symbol: None,
            indicator,
        }
    }

    fn normalize_symbol(&mut self) {
        if let Some(symbol) = &mut self.symbol {
            symbol.make_ascii_lowercase();
        }
    }

    fn canonical_cmp(&self, other: &Self) -> Ordering {
        self.symbol
            .cmp(&other.symbol)
            .then_with(|| {
                self.indicator
                    .canonical_name()
                    .cmp(other.indicator.canonical_name())
            })
            .then_with(|| self.indicator.identity_cmp(&other.indicator))
    }
}

/// Canonically ordered, serializable definitions for a complete extractor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FeatureSet {
    pub(crate) indicators: Vec<IndicatorDef>,
}

impl FeatureSet {
    pub fn new(mut indicators: Vec<IndicatorDef>) -> Self {
        for definition in &mut indicators {
            definition.normalize_symbol();
        }
        indicators.sort_by(IndicatorDef::canonical_cmp);
        Self { indicators }
    }

    /// Add one definition while preserving canonical feature-vector order.
    pub fn add_indicator(&mut self, mut definition: IndicatorDef) {
        definition.normalize_symbol();
        let index = self
            .indicators
            .partition_point(|existing| existing.canonical_cmp(&definition).is_le());
        self.indicators.insert(index, definition);
    }

    pub fn indicators(&self) -> &[IndicatorDef] {
        &self.indicators
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(source: ValueSource) -> IndicatorSpec {
        IndicatorSpec::Sma {
            source,
            windows: vec![2],
            warmup_policy: WarmupPolicy::FullWindow,
        }
    }

    #[test]
    fn feature_set_order_uses_scope_kind_and_identity() {
        let feature_set = FeatureSet::new(vec![
            IndicatorDef::symbol(
                "Z",
                IndicatorSpec::Ema {
                    source: ValueSource::Price,
                    windows: vec![2],
                    warmup_policy: WarmupPolicy::FullWindow,
                },
            ),
            IndicatorDef::symbol("A", sample(ValueSource::Volume)),
            IndicatorDef::global(IndicatorSpec::TimeSinceFirstEventOfDay {
                utc_offset_millis: 3_600_000,
            }),
            IndicatorDef::symbol("Z", sample(ValueSource::Price)),
            IndicatorDef::global(IndicatorSpec::DayOfWeek),
            IndicatorDef::symbol("A", sample(ValueSource::Price)),
        ]);

        let definitions = feature_set.indicators();
        assert!(matches!(definitions[0].indicator, IndicatorSpec::DayOfWeek));
        assert!(matches!(
            definitions[1].indicator,
            IndicatorSpec::TimeSinceFirstEventOfDay { .. }
        ));
        assert_eq!(definitions[2].symbol.as_deref(), Some("a"));
        assert!(matches!(
            definitions[2].indicator,
            IndicatorSpec::Sma {
                source: ValueSource::Price,
                ..
            }
        ));
        assert!(matches!(
            definitions[3].indicator,
            IndicatorSpec::Sma {
                source: ValueSource::Volume,
                ..
            }
        ));
        assert_eq!(definitions[4].symbol.as_deref(), Some("z"));
        assert_eq!(definitions[5].symbol.as_deref(), Some("z"));
        assert_eq!(definitions[4].indicator.canonical_name(), "ema");
        assert_eq!(definitions[5].indicator.canonical_name(), "sma");
    }
}

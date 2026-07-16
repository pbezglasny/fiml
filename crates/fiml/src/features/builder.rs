use std::time::Duration;

use crate::features::definition::{
    FeatureSet, IndicatorDef, IndicatorSpec, TimeWindows, ValueSource,
};

/// Fluent cold-path builder for a reusable [`FeatureSet`].
#[derive(Debug, Clone, Default)]
pub struct FeatureSetBuilder {
    feature_set: FeatureSet,
}

impl FeatureSetBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn indicator(mut self, definition: IndicatorDef) -> Self {
        self.feature_set.indicators.push(definition);
        self
    }

    pub fn sma(self, symbol: impl Into<String>, windows: impl IntoIterator<Item = usize>) -> Self {
        self.sma_from(symbol, ValueSource::Price, windows)
    }

    pub fn sma_from(
        self,
        symbol: impl Into<String>,
        source: ValueSource,
        windows: impl IntoIterator<Item = usize>,
    ) -> Self {
        self.indicator(IndicatorDef::symbol(
            symbol,
            IndicatorSpec::Sma {
                source,
                windows: windows.into_iter().collect(),
            },
        ))
    }

    pub fn ema(self, symbol: impl Into<String>, windows: impl IntoIterator<Item = usize>) -> Self {
        self.ema_from(symbol, ValueSource::Price, windows)
    }

    pub fn ema_from(
        self,
        symbol: impl Into<String>,
        source: ValueSource,
        windows: impl IntoIterator<Item = usize>,
    ) -> Self {
        self.indicator(IndicatorDef::symbol(
            symbol,
            IndicatorSpec::Ema {
                source,
                windows: windows.into_iter().collect(),
            },
        ))
    }

    pub fn sma_timed(
        self,
        symbol: impl Into<String>,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
    ) -> Self {
        self.sma_timed_from(symbol, ValueSource::Price, aggregation, windows)
    }

    pub fn sma_timed_from(
        self,
        symbol: impl Into<String>,
        source: ValueSource,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
    ) -> Self {
        self.indicator(IndicatorDef::symbol(
            symbol,
            IndicatorSpec::SmaTimed {
                source,
                time_windows: TimeWindows::new(aggregation, windows.into_iter().collect()),
            },
        ))
    }

    pub fn obv_timed(
        self,
        symbol: impl Into<String>,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
    ) -> Self {
        self.indicator(IndicatorDef::symbol(
            symbol,
            IndicatorSpec::ObvTimed {
                time_windows: TimeWindows::new(aggregation, windows.into_iter().collect()),
            },
        ))
    }

    pub fn trade_count_timed(
        self,
        symbol: impl Into<String>,
        aggregation: Duration,
        window: Duration,
    ) -> Self {
        self.indicator(IndicatorDef::symbol(
            symbol,
            IndicatorSpec::TradeCountTimed {
                aggregation,
                window,
            },
        ))
    }

    pub fn day_of_week(self) -> Self {
        self.indicator(IndicatorDef::global(IndicatorSpec::DayOfWeek))
    }

    pub fn time_since_first_event_of_day(self, utc_offset_millis: i64) -> Self {
        self.indicator(IndicatorDef::global(
            IndicatorSpec::TimeSinceFirstEventOfDay { utc_offset_millis },
        ))
    }

    pub fn build(self) -> FeatureSet {
        self.feature_set
    }
}

impl FeatureSet {
    pub fn builder() -> FeatureSetBuilder {
        FeatureSetBuilder::new()
    }
}

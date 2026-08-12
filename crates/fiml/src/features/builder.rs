use std::time::Duration;

use crate::features::definition::{
    FeatureSet, IndicatorSpec, ScopedIndicator, TimeWindows, ValueSource,
};
use crate::{Symbol, WarmupPolicy};

/// Fluent cold-path builder for a reusable [`FeatureSet`].
#[derive(Debug, Clone, Default)]
pub struct FeatureSetBuilder {
    indicators: Vec<ScopedIndicator>,
}

impl FeatureSetBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn indicator(mut self, definition: ScopedIndicator) -> Self {
        self.indicators.push(definition);
        self
    }

    pub fn sma(self, symbol: impl Into<Symbol>, windows: impl IntoIterator<Item = usize>) -> Self {
        self.sma_with_warmup(symbol, windows, WarmupPolicy::FullWindow)
    }

    pub fn sma_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        windows: impl IntoIterator<Item = usize>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.sma_from_with_warmup(symbol, ValueSource::Price, windows, warmup_policy)
    }

    pub fn sma_from(
        self,
        symbol: impl Into<Symbol>,
        source: ValueSource,
        windows: impl IntoIterator<Item = usize>,
    ) -> Self {
        self.sma_from_with_warmup(symbol, source, windows, WarmupPolicy::FullWindow)
    }

    pub fn sma_from_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        source: ValueSource,
        windows: impl IntoIterator<Item = usize>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.indicator(ScopedIndicator::symbol(
            symbol,
            IndicatorSpec::Sma {
                source,
                windows: windows.into_iter().collect(),
                warmup_policy,
            },
        ))
    }

    pub fn ema(self, symbol: impl Into<Symbol>, windows: impl IntoIterator<Item = usize>) -> Self {
        self.ema_with_warmup(symbol, windows, WarmupPolicy::FullWindow)
    }

    pub fn ema_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        windows: impl IntoIterator<Item = usize>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.ema_from_with_warmup(symbol, ValueSource::Price, windows, warmup_policy)
    }

    pub fn ema_from(
        self,
        symbol: impl Into<Symbol>,
        source: ValueSource,
        windows: impl IntoIterator<Item = usize>,
    ) -> Self {
        self.ema_from_with_warmup(symbol, source, windows, WarmupPolicy::FullWindow)
    }

    pub fn ema_from_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        source: ValueSource,
        windows: impl IntoIterator<Item = usize>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.indicator(ScopedIndicator::symbol(
            symbol,
            IndicatorSpec::Ema {
                source,
                windows: windows.into_iter().collect(),
                warmup_policy,
            },
        ))
    }

    pub fn cvd(self, symbol: impl Into<Symbol>, windows: impl IntoIterator<Item = usize>) -> Self {
        self.cvd_with_warmup(symbol, windows, WarmupPolicy::FullWindow)
    }

    pub fn cvd_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        windows: impl IntoIterator<Item = usize>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.indicator(ScopedIndicator::symbol(
            symbol,
            IndicatorSpec::Cvd {
                windows: windows.into_iter().collect(),
                warmup_policy,
            },
        ))
    }

    pub fn sma_timed(
        self,
        symbol: impl Into<Symbol>,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
    ) -> Self {
        self.sma_timed_with_warmup(symbol, aggregation, windows, WarmupPolicy::FullWindow)
    }

    pub fn sma_timed_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.sma_timed_from_with_warmup(
            symbol,
            ValueSource::Price,
            aggregation,
            windows,
            warmup_policy,
        )
    }

    pub fn sma_timed_from(
        self,
        symbol: impl Into<Symbol>,
        source: ValueSource,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
    ) -> Self {
        self.sma_timed_from_with_warmup(
            symbol,
            source,
            aggregation,
            windows,
            WarmupPolicy::FullWindow,
        )
    }

    pub fn sma_timed_from_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        source: ValueSource,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.indicator(ScopedIndicator::symbol(
            symbol,
            IndicatorSpec::SmaTimed {
                source,
                time_windows: TimeWindows::new(aggregation, windows.into_iter().collect()),
                warmup_policy,
            },
        ))
    }

    pub fn obv_timed(
        self,
        symbol: impl Into<Symbol>,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
    ) -> Self {
        self.obv_timed_with_warmup(symbol, aggregation, windows, WarmupPolicy::FullWindow)
    }

    pub fn obv_timed_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        aggregation: Duration,
        windows: impl IntoIterator<Item = Duration>,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.indicator(ScopedIndicator::symbol(
            symbol,
            IndicatorSpec::ObvTimed {
                time_windows: TimeWindows::new(aggregation, windows.into_iter().collect()),
                warmup_policy,
            },
        ))
    }

    pub fn trade_count_timed(
        self,
        symbol: impl Into<Symbol>,
        aggregation: Duration,
        window: Duration,
    ) -> Self {
        self.trade_count_timed_with_warmup(symbol, aggregation, window, WarmupPolicy::FullWindow)
    }

    pub fn trade_count_timed_with_warmup(
        self,
        symbol: impl Into<Symbol>,
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        self.indicator(ScopedIndicator::symbol(
            symbol,
            IndicatorSpec::TradeCountTimed {
                aggregation,
                window,
                warmup_policy,
            },
        ))
    }

    pub fn day_of_week(self) -> Self {
        self.indicator(ScopedIndicator::global(IndicatorSpec::DayOfWeek))
    }

    pub fn time_since_first_event_of_day(self, utc_offset_millis: i64) -> Self {
        self.indicator(ScopedIndicator::global(
            IndicatorSpec::TimeSinceFirstEventOfDay { utc_offset_millis },
        ))
    }

    pub fn build(self) -> FeatureSet {
        FeatureSet::new(self.indicators)
    }
}

impl FeatureSet {
    pub fn builder() -> FeatureSetBuilder {
        FeatureSetBuilder::new()
    }
}

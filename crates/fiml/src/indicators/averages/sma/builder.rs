use std::time::Duration;

use crate::features::EventKind;
use crate::features::builtin::sma::{self, MAX_WINDOWS_PER_SMA};
use crate::indicators::builder::{IndicatorFeatureVectorBuilder, PendingFeature};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, Result, Symbol};

#[derive(Clone, Copy)]
pub(crate) struct PendingSmaPeriods {
    pub(crate) symbol: Symbol,
    pub(crate) event_kind: EventKind,
    pub(crate) periods: [usize; MAX_WINDOWS_PER_SMA],
    pub(crate) window_count: usize,
    pub(crate) max_period: usize,
    pub(crate) output_start: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct PendingSmaTimedPeriods {
    pub(crate) symbol: Symbol,
    pub(crate) aggregation: Duration,
    pub(crate) periods: [usize; MAX_WINDOWS_PER_SMA],
    pub(crate) window_count: usize,
    pub(crate) max_period: usize,
    pub(crate) output_start: usize,
}

/// Nested builder for a sample-period SMA indicator.
pub struct SmaPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    symbol: Symbol,
    event_kind: EventKind,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
}

/// Nested builder for a time-bucketed SMA indicator.
pub struct SmaTimedPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    symbol: Symbol,
    aggregation: Duration,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
}

impl<F, V, const M: usize> SmaPeriodsBuilder<F, V, M, false>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new(parent: IndicatorFeatureVectorBuilder<F, V, M>, symbol: Symbol) -> Self {
        Self {
            parent,
            symbol,
            event_kind: EventKind::Price,
            periods: [0; MAX_WINDOWS_PER_SMA],
            window_count: 0,
            max_period: 0,
        }
    }

    /// Add the first sample-period SMA window.
    pub fn window(mut self, period: usize) -> Result<SmaPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(SmaPeriodsBuilder {
            parent: self.parent,
            symbol: self.symbol,
            event_kind: self.event_kind,
            periods: self.periods,
            window_count: self.window_count,
            max_period: self.max_period,
        })
    }
}

impl<F, V, const M: usize> SmaPeriodsBuilder<F, V, M, true>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Add another sample-period SMA window.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the SMA indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::SmaPeriods(PendingSmaPeriods {
                periods: self.periods,
                symbol: self.symbol,
                event_kind: self.event_kind,
                window_count: self.window_count,
                max_period: self.max_period,
                output_start,
            }));
        Ok(self.parent)
    }
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> SmaPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Set the event kind this SMA consumes. Defaults to [`EventKind::Price`].
    pub fn event_kind(mut self, event_kind: EventKind) -> Result<Self> {
        sma::validate_event_kind(event_kind)?;
        self.event_kind = event_kind;
        Ok(self)
    }

    fn push_window(&mut self, period: usize) -> Result<()> {
        sma::validate_period(period)?;
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_SMA, "SMA")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        self.max_period = self.max_period.max(period);
        Ok(())
    }
}

impl<F, V, const M: usize> SmaTimedPeriodsBuilder<F, V, M, false>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new(
        parent: IndicatorFeatureVectorBuilder<F, V, M>,
        symbol: Symbol,
        aggregation: Duration,
    ) -> Self {
        Self {
            parent,
            symbol,
            aggregation,
            periods: [0; MAX_WINDOWS_PER_SMA],
            window_count: 0,
            max_period: 0,
        }
    }

    /// Add the first timed SMA window, measured in aggregation buckets.
    pub fn window(mut self, period: usize) -> Result<SmaTimedPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(SmaTimedPeriodsBuilder {
            parent: self.parent,
            symbol: self.symbol,
            aggregation: self.aggregation,
            periods: self.periods,
            window_count: self.window_count,
            max_period: self.max_period,
        })
    }
}

impl<F, V, const M: usize> SmaTimedPeriodsBuilder<F, V, M, true>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Add another timed SMA window, measured in aggregation buckets.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the timed SMA indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::SmaTimedPeriods(PendingSmaTimedPeriods {
                aggregation: self.aggregation,
                symbol: self.symbol,
                periods: self.periods,
                window_count: self.window_count,
                max_period: self.max_period,
                output_start,
            }));
        Ok(self.parent)
    }
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> SmaTimedPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        if self.aggregation.as_millis() == 0 {
            return Err(FimlError::InvalidArgument(
                "SMA timed aggregation must be at least 1 millisecond".to_string(),
            ));
        }
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                "SMA timed period must be at least 1".to_string(),
            ));
        }
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_SMA, "SMA timed")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        self.max_period = self.max_period.max(period);
        Ok(())
    }
}

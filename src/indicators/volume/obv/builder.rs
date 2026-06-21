use std::time::Duration;

use crate::features::builtin::obv::{self, MAX_WINDOWS_PER_OBV};
use crate::indicators::builder::{IndicatorFeatureVectorBuilder, PendingFeature};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, Result, Symbol};

#[derive(Clone, Copy)]
pub(crate) struct PendingObvTimedPeriods {
    pub(crate) symbol: Symbol,
    pub(crate) aggregation: Duration,
    pub(crate) periods: [usize; MAX_WINDOWS_PER_OBV],
    pub(crate) window_count: usize,
    pub(crate) max_period: usize,
    pub(crate) output_start: usize,
}

/// Nested builder for a time-bucketed OBV indicator.
pub struct ObvTimedPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    symbol: Symbol,
    aggregation: Duration,
    periods: [usize; MAX_WINDOWS_PER_OBV],
    window_count: usize,
    max_period: usize,
}

impl<F, V, const M: usize> ObvTimedPeriodsBuilder<F, V, M, false>
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
            periods: [0; MAX_WINDOWS_PER_OBV],
            window_count: 0,
            max_period: 0,
        }
    }

    /// Add the first timed OBV window, measured in aggregation buckets.
    pub fn window(mut self, period: usize) -> Result<ObvTimedPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(ObvTimedPeriodsBuilder {
            parent: self.parent,
            symbol: self.symbol,
            aggregation: self.aggregation,
            periods: self.periods,
            window_count: self.window_count,
            max_period: self.max_period,
        })
    }
}

impl<F, V, const M: usize> ObvTimedPeriodsBuilder<F, V, M, true>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Add another timed OBV window, measured in aggregation buckets.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the timed OBV indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::ObvTimedPeriods(PendingObvTimedPeriods {
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

impl<F, V, const M: usize, const HAS_WINDOWS: bool> ObvTimedPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        if self.aggregation.as_millis() == 0 {
            return Err(FimlError::InvalidArgument(
                "OBV timed aggregation must be at least 1 millisecond".to_string(),
            ));
        }
        obv::validate_period(period)?;
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_OBV, "OBV timed")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        self.max_period = self.max_period.max(period);
        Ok(())
    }
}

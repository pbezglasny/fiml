use crate::builder::{IndicatorFeatureVectorBuilder, PendingFeature};
use crate::features::builtin::ema::{self, MAX_WINDOWS_PER_EMA};
use crate::vectors::FeatureOutput;
use crate::{Float, Result, Ticker};

#[derive(Clone, Copy)]
pub(crate) struct PendingEmaPeriods {
    pub(crate) ticker: Ticker,
    pub(crate) periods: [usize; MAX_WINDOWS_PER_EMA],
    pub(crate) window_count: usize,
    pub(crate) output_start: usize,
}

/// Nested builder for a sample-period EMA indicator.
pub struct EmaPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float,
    V: FeatureOutput<F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    ticker: Ticker,
    periods: [usize; MAX_WINDOWS_PER_EMA],
    window_count: usize,
}

impl<F, V, const M: usize> EmaPeriodsBuilder<F, V, M, false>
where
    F: Float,
    V: FeatureOutput<F>,
{
    pub(crate) fn new(parent: IndicatorFeatureVectorBuilder<F, V, M>, ticker: Ticker) -> Self {
        Self {
            parent,
            ticker,
            periods: [0; MAX_WINDOWS_PER_EMA],
            window_count: 0,
        }
    }

    /// Add the first sample-period EMA window.
    pub fn window(mut self, period: usize) -> Result<EmaPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(EmaPeriodsBuilder {
            parent: self.parent,
            ticker: self.ticker,
            periods: self.periods,
            window_count: self.window_count,
        })
    }
}

impl<F, V, const M: usize> EmaPeriodsBuilder<F, V, M, true>
where
    F: Float,
    V: FeatureOutput<F>,
{
    /// Add another sample-period EMA window.
    pub fn window(mut self, period: usize) -> Result<Self> {
        self.push_window(period)?;
        Ok(self)
    }

    /// Finish the EMA indicator and return to the parent feature-vector builder.
    pub fn done(mut self) -> Result<IndicatorFeatureVectorBuilder<F, V, M>> {
        let output_start = self.parent.reserve_outputs(self.window_count)?;
        self.parent
            .push_entry(PendingFeature::EmaPeriods(PendingEmaPeriods {
                periods: self.periods,
                ticker: self.ticker,
                window_count: self.window_count,
                output_start,
            }));
        Ok(self.parent)
    }
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> EmaPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float,
    V: FeatureOutput<F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        ema::validate_period(period)?;
        self.parent
            .ensure_can_push_window(self.window_count, MAX_WINDOWS_PER_EMA, "EMA")?;

        self.periods[self.window_count] = period;
        self.window_count += 1;
        Ok(())
    }
}

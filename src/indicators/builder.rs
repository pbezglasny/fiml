use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::time::Duration;

use crate::features::builtin::BuiltinFeature;
use crate::features::builtin::{day_of_week, ema as ema_indicator, sma as sma_indicator};
use crate::features::indicator_vector::IndicatorFeatureVector;
use crate::indicators::averages::{
    EmaPeriodsBuilder, PendingEmaPeriods, PendingSmaPeriods, PendingSmaTimedPeriods,
    SmaPeriodsBuilder, SmaTimedPeriodsBuilder,
};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, Result, Symbol};

pub(crate) enum PendingFeature {
    SmaPeriods(PendingSmaPeriods),
    EmaPeriods(PendingEmaPeriods),
    SmaTimedPeriods(PendingSmaTimedPeriods),
    DayOfWeek { ticker: Symbol, output_index: usize },
}

/// Fixed-capacity builder for [`IndicatorFeatureVector`] instances backed by
/// library-provided builtin features.
pub struct IndicatorFeatureVectorBuilder<F, V, const M: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    cells: V,
    cell_capacity: usize,
    entries: [MaybeUninit<PendingFeature>; M],
    entry_count: usize,
    output_count: usize,
    _marker: PhantomData<F>,
}

impl<F, V, const M: usize> IndicatorFeatureVectorBuilder<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Start building a feature vector that writes into `cells`.
    pub fn new(cells: V) -> Self {
        let cell_capacity = cells.len();
        Self {
            cells,
            cell_capacity,
            entries: [const { MaybeUninit::uninit() }; M],
            entry_count: 0,
            output_count: 0,
            _marker: PhantomData,
        }
    }

    /// Configure a sample-period SMA indicator.
    pub fn sma_periods(self, ticker: Symbol) -> SmaPeriodsBuilder<F, V, M, false> {
        SmaPeriodsBuilder::new(self, ticker)
    }

    /// Configure a sample-period EMA indicator.
    pub fn ema_periods(self, ticker: Symbol) -> EmaPeriodsBuilder<F, V, M, false> {
        EmaPeriodsBuilder::new(self, ticker)
    }

    /// Configure a time-bucketed SMA indicator.
    ///
    /// Price event timestamps are passed directly to the timed SMA and must be
    /// milliseconds, matching the indicator's `Duration::as_millis()` windows.
    pub fn sma_timed(
        self,
        ticker: Symbol,
        aggregation: Duration,
    ) -> SmaTimedPeriodsBuilder<F, V, M, false> {
        SmaTimedPeriodsBuilder::new(self, ticker, aggregation)
    }

    /// Add a day-of-week output cell.
    pub fn day_of_week(mut self, ticker: Symbol) -> Result<Self> {
        let output_index = self.reserve_outputs(1)?;
        self.push_entry(PendingFeature::DayOfWeek {
            ticker,
            output_index,
        });
        Ok(self)
    }

    /// Finish the builder and return the dispatchable feature vector.
    pub fn build(self) -> Result<IndicatorFeatureVector<F, V, BuiltinFeature<F>, M>> {
        let mut entries = [const { MaybeUninit::uninit() }; M];
        let mut names = vec![None; self.cell_capacity].into_boxed_slice();

        for (entry_index, pending) in self.pending_entries().enumerate() {
            let entry = match pending {
                PendingFeature::SmaPeriods(config) => {
                    sma_indicator::build_sma_periods_entry(config, &mut names)
                }
                PendingFeature::EmaPeriods(config) => {
                    ema_indicator::build_ema_periods_entry(config, &mut names)
                }
                PendingFeature::SmaTimedPeriods(config) => {
                    sma_indicator::build_sma_timed_periods_entry(config, &mut names)?
                }
                PendingFeature::DayOfWeek {
                    ticker,
                    output_index,
                } => day_of_week::build_entry(*ticker, *output_index, &mut names),
            };
            entries[entry_index].write(entry);
        }

        Ok(IndicatorFeatureVector::from_builtin_entries(
            self.cells,
            entries,
            self.entry_count,
            names,
        ))
    }

    fn pending_entries(&self) -> impl Iterator<Item = &PendingFeature> {
        self.entries
            .iter()
            .take(self.entry_count)
            .map(|entry| unsafe { entry.assume_init_ref() })
    }

    pub(crate) fn ensure_can_push_window(
        &self,
        window_count: usize,
        max_windows: usize,
        indicator_name: &str,
    ) -> Result<()> {
        if self.entry_count >= M {
            return Err(FimlError::InvalidArgument(format!(
                "too many feature instances: capacity is {M}"
            )));
        }
        if window_count >= max_windows {
            return Err(FimlError::InvalidArgument(format!(
                "too many {indicator_name} windows: capacity is {max_windows}"
            )));
        }
        let needed_outputs = self.output_count + window_count + 1;
        if needed_outputs > self.cell_capacity {
            return Err(FimlError::InvalidArgument(format!(
                "too many output cells: {needed_outputs} (capacity: {})",
                self.cell_capacity
            )));
        }

        Ok(())
    }

    pub(crate) fn reserve_outputs(&mut self, count: usize) -> Result<usize> {
        if self.entry_count >= M {
            return Err(FimlError::InvalidArgument(format!(
                "too many feature instances: capacity is {M}"
            )));
        }
        if self.output_count + count > self.cell_capacity {
            return Err(FimlError::InvalidArgument(format!(
                "too many output cells: {} (capacity: {})",
                self.output_count + count,
                self.cell_capacity
            )));
        }

        let output_start = self.output_count;
        self.output_count += count;
        Ok(output_start)
    }

    pub(crate) fn push_entry(&mut self, entry: PendingFeature) {
        self.entries[self.entry_count].write(entry);
        self.entry_count += 1;
    }
}

impl<F, V, const M: usize> Default for IndicatorFeatureVectorBuilder<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F> + Default,
{
    fn default() -> Self {
        Self::new(V::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{IndicatorFeatures, MAX_WINDOWS_PER_SMA, SmaPeriodsBuilder};
    use crate::{ArrayFeatureVector, Event, EventKind, ticker};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn builds_single_sma_period_window() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods(aapl)
                .window(2)?
                .done()?
                .build()?;

        fv.dispatch(&Event::price(aapl, 10.0, 0));
        fv.dispatch(&Event::price(aapl, 20.0, 0));

        assert!(approx_eq(fv.feature_vector().values()[0], 15.0));
        assert_eq!(fv.index_of(aapl, "sma_periods_2"), Some(0));
        Ok(())
    }

    #[test]
    fn one_sma_feature_writes_multiple_period_windows() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 2>::new())
                .sma_periods(aapl)
                .window(2)?
                .window(5)?
                .done()?
                .build()?;

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 4.5));
        assert!(approx_eq(fv.feature_vector().values()[1], 3.0));
        assert_eq!(fv.index_of(aapl, "sma_periods_2"), Some(0));
        assert_eq!(fv.index_of(aapl, "sma_periods_5"), Some(1));
        Ok(())
    }

    #[test]
    fn sma_periods_can_subscribe_to_volume_events() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 2>::new(ArrayFeatureVector::<f64, 2>::new())
                .sma_periods(aapl)
                .window(3)?
                .done()?
                .sma_periods(aapl)
                .event_kind(EventKind::Volume)?
                .window(3)?
                .done()?
                .build()?;

        for v in [3.0, 6.0, 9.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }
        for v in [100.0, 200.0, 300.0] {
            fv.dispatch(&Event::volume(aapl, v, 0));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 6.0));
        assert!(approx_eq(fv.feature_vector().values()[1], 200.0));
        assert_eq!(fv.index_of(aapl, "sma_periods_3"), Some(0));
        assert_eq!(fv.index_of(aapl, "volume_sma_periods_3"), Some(1));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_sma_event_kind() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods(aapl)
                .event_kind(EventKind::OrderBook);

        assert!(built.is_err());
    }

    #[test]
    fn one_ema_feature_writes_multiple_period_windows() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 2>::new())
                .ema_periods(aapl)
                .window(3)?
                .window(5)?
                .done()?
                .build()?;

        for v in [10.0, 20.0, 30.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 22.5));
        assert!(approx_eq(
            fv.feature_vector().values()[1],
            18.888888888888886
        ));
        assert_eq!(fv.index_of(aapl, "ema_periods_3"), Some(0));
        assert_eq!(fv.index_of(aapl, "ema_periods_5"), Some(1));
        Ok(())
    }

    #[test]
    fn ema_periods_can_subscribe_to_volume_events() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 2>::new(ArrayFeatureVector::<f64, 2>::new())
                .ema_periods(aapl)
                .window(3)?
                .done()?
                .ema_periods(aapl)
                .event_kind(EventKind::Volume)?
                .window(3)?
                .done()?
                .build()?;

        for v in [10.0, 20.0, 30.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }
        for v in [100.0, 200.0, 300.0] {
            fv.dispatch(&Event::volume(aapl, v, 0));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 22.5));
        assert!(approx_eq(fv.feature_vector().values()[1], 225.0));
        assert_eq!(fv.index_of(aapl, "ema_periods_3"), Some(0));
        assert_eq!(fv.index_of(aapl, "volume_ema_periods_3"), Some(1));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_ema_event_kind() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .ema_periods(aapl)
                .event_kind(EventKind::Time);

        assert!(built.is_err());
    }

    #[test]
    fn one_sma_timed_feature_writes_multiple_period_windows() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 2>::new())
                .sma_timed(aapl, Duration::from_millis(1_000))
                .window(2)?
                .window(3)?
                .done()?
                .build()?;

        for (value, timestamp) in [
            (10.0, 0),
            (20.0, 1_000),
            (30.0, 2_000),
            (40.0, 2_500),
            (50.0, 3_000),
        ] {
            fv.dispatch(&Event::price(aapl, value, timestamp));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 42.5));
        assert!(approx_eq(fv.feature_vector().values()[1], 35.0));
        assert_eq!(fv.index_of(aapl, "sma_timed_periods_2"), Some(0));
        assert_eq!(fv.index_of(aapl, "sma_timed_periods_3"), Some(1));
        Ok(())
    }

    #[test]
    fn rejects_zero_sma_timed_period() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_timed(aapl, Duration::from_millis(1_000))
                .window(0);

        assert!(built.is_err());
    }

    #[test]
    fn rejects_zero_sma_timed_aggregation() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_timed(aapl, Duration::ZERO)
                .window(1);

        assert!(built.is_err());
    }

    #[test]
    fn chains_day_of_week_after_sma_periods() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 2>::new(ArrayFeatureVector::<f64, 2>::new())
                .sma_periods(aapl)
                .window(3)?
                .done()?
                .day_of_week(aapl)?
                .build()?;

        for v in [3.0, 6.0, 9.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }
        fv.dispatch(&Event::time(1_609_459_200));

        assert!(approx_eq(fv.feature_vector().values()[0], 6.0));
        assert!(approx_eq(fv.feature_vector().values()[1], 5.0));
        assert_eq!(fv.index_of(aapl, "day_of_week"), Some(1));
        Ok(())
    }

    #[test]
    fn rejects_zero_sma_period() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods(aapl)
                .window(0);

        assert!(built.is_err());
    }

    #[test]
    fn rejects_zero_ema_period() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .ema_periods(aapl)
                .window(0);

        assert!(built.is_err());
    }

    #[test]
    fn rejects_more_output_cells_than_capacity() {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods(aapl)
                .window(2)
                .unwrap()
                .window(5);

        assert!(built.is_err());
    }

    #[test]
    fn rejects_more_feature_instances_than_capacity() -> Result<()> {
        let aapl = ticker::intern("AAPL");
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 2>::new())
                .day_of_week(aapl)?
                .day_of_week(aapl);

        assert!(built.is_err());
        Ok(())
    }

    #[test]
    fn rejects_more_sma_windows_than_capacity() {
        let aapl = ticker::intern("AAPL");
        let mut builder = IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<
            f64,
            { MAX_WINDOWS_PER_SMA + 1 },
        >::new())
        .sma_periods(aapl)
        .window(1)
        .unwrap();

        for period in 2..=MAX_WINDOWS_PER_SMA {
            builder = builder.window(period).unwrap();
        }

        assert!(builder.window(MAX_WINDOWS_PER_SMA + 1).is_err());
    }

    #[test]
    fn root_reexports_are_usable() -> crate::Result<()> {
        use crate::{IndicatorFeatureVector, IndicatorFeatureVectorBuilder};

        let aapl = ticker::intern("AAPL");
        let fv: IndicatorFeatureVector<_, _, BuiltinFeature<f64>, 1> =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods(aapl)
                .window(2)?
                .done()?
                .build()?;

        assert_eq!(fv.index_of(aapl, "sma_periods_2"), Some(0));

        let _: Option<SmaPeriodsBuilder<f64, ArrayFeatureVector<f64, 1>, 1, false>> = None;
        Ok(())
    }
}

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::features::builtin::{BuiltinFeature, DayOfWeek, MAX_WINDOWS_PER_SMA};
use crate::features::event::EventKind;
use crate::features::vector::{BuiltinFeatureEntry, IndicatorFeatureVector};
use crate::vectors::FeatureOutput;
use crate::{FimlError, Float, HeapRingBuffer, Result, SimpleMovingAverage};

#[derive(Clone, Copy)]
struct PendingSmaPeriods {
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
    output_start: usize,
}

#[derive(Clone, Copy)]
enum PendingFeature {
    SmaPeriods(PendingSmaPeriods),
    DayOfWeek { output_index: usize },
}

/// Fixed-capacity builder for [`IndicatorFeatureVector`] instances backed by
/// library-provided builtin features.
pub struct IndicatorFeatureVectorBuilder<F, V, const M: usize>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
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
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    /// Start building a feature vector that writes into `cells`.
    pub fn new(cells: V) -> Self {
        let cell_capacity = cells.capacity();
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
    pub fn sma_periods(self) -> SmaPeriodsBuilder<F, V, M, false> {
        SmaPeriodsBuilder {
            parent: self,
            periods: [0; MAX_WINDOWS_PER_SMA],
            window_count: 0,
            max_period: 0,
        }
    }

    /// Add a day-of-week output cell.
    pub fn day_of_week(mut self) -> Result<Self> {
        let output_index = self.reserve_outputs(1)?;
        self.push_entry(PendingFeature::DayOfWeek { output_index });
        Ok(self)
    }

    /// Finish the builder and return the dispatchable feature vector.
    pub fn build(self) -> Result<IndicatorFeatureVector<F, V, BuiltinFeature<F>, M>> {
        let mut entries = [const { MaybeUninit::uninit() }; M];
        let mut names = vec![None; self.cell_capacity].into_boxed_slice();

        for (entry_index, pending) in self.pending_entries().enumerate() {
            let entry = match pending {
                PendingFeature::SmaPeriods(config) => build_sma_periods_entry(config, &mut names),
                PendingFeature::DayOfWeek { output_index } => {
                    names[*output_index] = Some("day_of_week".to_string());
                    BuiltinFeatureEntry {
                        feature: BuiltinFeature::DayOfWeek(DayOfWeek::new(*output_index)),
                        kind: EventKind::Time,
                    }
                }
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

    fn reserve_outputs(&mut self, count: usize) -> Result<usize> {
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

    fn push_entry(&mut self, entry: PendingFeature) {
        self.entries[self.entry_count].write(entry);
        self.entry_count += 1;
    }
}

impl<F, V, const M: usize> Default for IndicatorFeatureVectorBuilder<F, V, M>
where
    F: Float + 'static,
    V: FeatureOutput<F> + Default,
{
    fn default() -> Self {
        Self::new(V::default())
    }
}

/// Nested builder for a sample-period SMA indicator.
pub struct SmaPeriodsBuilder<F, V, const M: usize, const HAS_WINDOWS: bool>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    parent: IndicatorFeatureVectorBuilder<F, V, M>,
    periods: [usize; MAX_WINDOWS_PER_SMA],
    window_count: usize,
    max_period: usize,
}

impl<F, V, const M: usize, const HAS_WINDOWS: bool> SmaPeriodsBuilder<F, V, M, HAS_WINDOWS>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    fn push_window(&mut self, period: usize) -> Result<()> {
        if period == 0 {
            return Err(FimlError::InvalidArgument(
                "SMA period must be at least 1".to_string(),
            ));
        }
        if self.parent.entry_count >= M {
            return Err(FimlError::InvalidArgument(format!(
                "too many feature instances: capacity is {M}"
            )));
        }
        if self.window_count >= MAX_WINDOWS_PER_SMA {
            return Err(FimlError::InvalidArgument(format!(
                "too many SMA windows: capacity is {MAX_WINDOWS_PER_SMA}"
            )));
        }
        let needed_outputs = self.parent.output_count + self.window_count + 1;
        if needed_outputs > self.parent.cell_capacity {
            return Err(FimlError::InvalidArgument(format!(
                "too many output cells: {needed_outputs} (capacity: {})",
                self.parent.cell_capacity
            )));
        }

        self.periods[self.window_count] = period;
        self.window_count += 1;
        self.max_period = self.max_period.max(period);
        Ok(())
    }
}

impl<F, V, const M: usize> SmaPeriodsBuilder<F, V, M, false>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
{
    /// Add the first sample-period SMA window.
    pub fn window(mut self, period: usize) -> Result<SmaPeriodsBuilder<F, V, M, true>> {
        self.push_window(period)?;
        Ok(SmaPeriodsBuilder {
            parent: self.parent,
            periods: self.periods,
            window_count: self.window_count,
            max_period: self.max_period,
        })
    }
}

impl<F, V, const M: usize> SmaPeriodsBuilder<F, V, M, true>
where
    F: Float + 'static,
    V: FeatureOutput<F>,
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
                window_count: self.window_count,
                max_period: self.max_period,
                output_start,
            }));
        Ok(self.parent)
    }
}

fn build_sma_periods_entry<F: Float + 'static>(
    config: &PendingSmaPeriods,
    names: &mut [Option<String>],
) -> BuiltinFeatureEntry<F> {
    let mut sma = SimpleMovingAverage::<HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>::new_heap(
        config.max_period,
    );
    let mut output_indexes = [0; MAX_WINDOWS_PER_SMA];

    for (window_index, period) in config
        .periods
        .iter()
        .copied()
        .enumerate()
        .take(config.window_count)
    {
        sma.add_window(period)
            .expect("validated SMA period should fit its ring buffer");
        let output_index = config.output_start + window_index;
        output_indexes[window_index] = output_index;
        names[output_index] = Some(format!("sma_periods_{period}"));
    }

    BuiltinFeatureEntry {
        feature: BuiltinFeature::Sma {
            sma,
            output_indexes,
            output_count: config.window_count,
        },
        kind: EventKind::Price,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, Event};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn builds_single_sma_period_window() -> Result<()> {
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods()
                .window(2)?
                .done()?
                .build()?;

        fv.dispatch(&Event::price(10.0, 0));
        fv.dispatch(&Event::price(20.0, 0));

        assert!(approx_eq(fv.values()[0], 15.0));
        assert_eq!(fv.index_of("sma_periods_2"), Some(0));
        Ok(())
    }

    #[test]
    fn one_sma_feature_writes_multiple_period_windows() -> Result<()> {
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 2>::new())
                .sma_periods()
                .window(2)?
                .window(5)?
                .done()?
                .build()?;

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            fv.dispatch(&Event::price(v, 0));
        }

        assert!(approx_eq(fv.values()[0], 4.5));
        assert!(approx_eq(fv.values()[1], 3.0));
        assert_eq!(fv.index_of("sma_periods_2"), Some(0));
        assert_eq!(fv.index_of("sma_periods_5"), Some(1));
        Ok(())
    }

    #[test]
    fn chains_day_of_week_after_sma_periods() -> Result<()> {
        let mut fv =
            IndicatorFeatureVectorBuilder::<f64, _, 2>::new(ArrayFeatureVector::<f64, 2>::new())
                .sma_periods()
                .window(3)?
                .done()?
                .day_of_week()?
                .build()?;

        for v in [3.0, 6.0, 9.0] {
            fv.dispatch(&Event::price(v, 0));
        }
        fv.dispatch(&Event::time(1_609_459_200));

        assert!(approx_eq(fv.values()[0], 6.0));
        assert!(approx_eq(fv.values()[1], 5.0));
        assert_eq!(fv.index_of("day_of_week"), Some(1));
        Ok(())
    }

    #[test]
    fn rejects_zero_sma_period() {
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods()
                .window(0);

        assert!(built.is_err());
    }

    #[test]
    fn rejects_more_output_cells_than_capacity() {
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods()
                .window(2)
                .unwrap()
                .window(5);

        assert!(built.is_err());
    }

    #[test]
    fn rejects_more_feature_instances_than_capacity() -> Result<()> {
        let built =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 2>::new())
                .day_of_week()?
                .day_of_week();

        assert!(built.is_err());
        Ok(())
    }

    #[test]
    fn rejects_more_sma_windows_than_capacity() {
        let mut builder = IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<
            f64,
            { MAX_WINDOWS_PER_SMA + 1 },
        >::new())
        .sma_periods()
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

        let fv: IndicatorFeatureVector<_, _, BuiltinFeature<f64>, 1> =
            IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
                .sma_periods()
                .window(2)?
                .done()?
                .build()?;

        assert_eq!(fv.index_of("sma_periods_2"), Some(0));
        Ok(())
    }
}

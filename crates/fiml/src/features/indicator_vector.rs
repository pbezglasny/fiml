use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::features::builtin::BuiltinFeature;
use crate::features::builtin::{day_of_week, ema, obv, sma};
use crate::features::event::{EVENT_KIND_COUNT, Event, EventKind};
use crate::features::spec::BuiltinSpec;
use crate::symbols::resolve;
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, Result, Symbol};

/// Contract every feature implements.
///
/// A feature subscribes to exactly one [`EventKind`](crate::features::EventKind)
/// and the feature vector only hands it events of that kind, so `update` reacts
/// to its own variant and ignores the rest. Computed values are written by
/// output index through the feature vector passed to `update`.
/// Implementations are dispatched statically (via enums), so every call
/// monomorphizes to a direct function call.
pub trait Feature<F: Float> {
    fn update<O: FeatureVector<F = F>>(&mut self, event: &Event<F>, output: &mut O);
}

pub trait IndicatorFeatures {
    type F: Float;
    type FeatureVector: FeatureVector<F = Self::F>;

    fn feature_vector(&self) -> &Self::FeatureVector;

    /// Route an event to the features subscribing to its kind, writing fresh
    /// values into their cells. Features of other kinds are not touched.
    fn dispatch(&mut self, event: &Event<Self::F>);

    /// Cell index a named feature for `symbol` writes to, or `None` if no such key.
    fn index_of(&self, symbol: Symbol, name: &str) -> Option<usize>;
}

#[derive(Clone)]
pub(crate) struct FeatureKey {
    pub(crate) symbol: Symbol,
    pub(crate) name: String,
}

/// Self-contained feature vector: owns the output `cells` and the `features`
/// that write into them, and routes each incoming [`Event`] to only the
/// features that subscribe to its kind.
///
/// The cell storage is any [`FeatureVector`] implementation `V`, so the cell
/// count can be fixed at compile time (`ArrayFeatureVector`) or chosen at
/// runtime by a heap-backed implementation
/// Features are stored grouped by [`EventKind`](crate::features::EventKind):
/// `groups[k]` is the `(start, len)` slice of `features` subscribing to kind
/// `k`, so [`dispatch`](Self::dispatch) iterates only the relevant group.
///
/// Features store output cell indexes, not references into the cell storage.
/// During dispatch, the feature receives mutable access to the output storage and
/// writes by index. This keeps the aggregate movable without self-references.
///
/// - `V` — cell storage, any [`FeatureVector<Float = F>`].
/// - `M` — capacity of the feature array.
pub(crate) struct BuiltinFeatureEntry<F: Float> {
    pub(crate) feature: BuiltinFeature<F>,
    pub(crate) kind: EventKind,
}

pub struct IndicatorFeatureVector<F, V, I, const M: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: Feature<F>,
{
    feature_vector: V,
    features: [MaybeUninit<I>; M],
    feature_count: usize,
    groups: [(usize, usize); EVENT_KIND_COUNT],
    names: Box<[Option<FeatureKey>]>,
    _marker: PhantomData<F>,
}

impl<F, V, I, const M: usize> IndicatorFeatureVector<F, V, I, M>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: Feature<F>,
{
    pub fn feature_vector(&self) -> &V {
        &self.feature_vector
    }

    /// Feature names in output-cell order, so callers can label each column of
    /// [`feature_vector().values()`](Self::feature_vector). Cells without a
    /// registered feature yield an empty string.
    pub fn feature_names(&self) -> Vec<String> {
        self.names
            .iter()
            .map(|key| match key {
                Some(key) => key.name.clone(),
                None => String::new(),
            })
            .collect()
    }
}

impl<F, V, I, const M: usize> IndicatorFeatures for IndicatorFeatureVector<F, V, I, M>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: Feature<F>,
{
    type F = F;
    type FeatureVector = V;

    fn feature_vector(&self) -> &Self::FeatureVector {
        &self.feature_vector
    }

    fn dispatch(&mut self, event: &Event<F>) {
        let (start, len) = self.groups[event.kind() as usize];
        // SAFETY: every slot in `features[..feature_count]` is initialized, and
        // each group is a sub-range of that, so this slice is initialized.
        let features = &mut self.features[start..start + len];
        let cells = &mut self.feature_vector;
        for slot in features {
            let feature = unsafe { slot.assume_init_mut() };
            feature.update(event, cells);
        }
    }

    fn index_of(&self, symbol: Symbol, name: &str) -> Option<usize> {
        self.names
            .iter()
            .position(|n| matches!(n, Some(n) if n.symbol == symbol && n.name == name))
    }
}

impl<F, V, I, const M: usize> Drop for IndicatorFeatureVector<F, V, I, M>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: Feature<F>,
{
    fn drop(&mut self) {
        // SAFETY: the first `feature_count` entries are initialized (groups
        // partition `0..feature_count` with no gaps).
        for slot in &mut self.features[..self.feature_count] {
            unsafe { slot.assume_init_drop() };
        }
    }
}

impl<F, V, const M: usize> IndicatorFeatureVector<F, V, BuiltinFeature<F>, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Build a feature vector from library builtins, one feature per spec
    /// (per-spec construction: each SMA gets its own ring buffer).
    ///
    /// `cells` is the output storage; it is taken by value so the caller can
    /// size it however it likes (compile-time or runtime). `specs` pairs each
    /// output's name and symbol with its [`BuiltinSpec`]. The `(symbol, name)`
    /// key is recorded against the cell the feature writes to (cells are
    /// assigned in spec order), so
    /// [`index_of`] and [`values`] keep the caller's order, while the features
    /// themselves are stored grouped by event kind for routing.
    ///
    /// [`index_of`]: Self::index_of
    /// [`values`]: Self::values
    pub fn from_builtin_specs(cells: V, specs: &[(&str, Symbol, BuiltinSpec)]) -> Result<Self> {
        let cell_count = cells.len();
        if specs.len() > M || specs.len() > cell_count {
            return Err(FimlError::InvalidArgument(format!(
                "too many features: {} (cells: {cell_count}, capacity: {M})",
                specs.len()
            )));
        }
        validate_builtin_specs(specs)?;

        let mut entries = [const { MaybeUninit::uninit() }; M];
        let mut names = vec![None; cell_count].into_boxed_slice();

        for (output_index, (name, symbol, spec)) in specs.iter().enumerate() {
            entries[output_index].write(BuiltinFeatureEntry {
                feature: build_builtin(*symbol, spec, output_index)?,
                kind: spec.event_kind(),
            });
            names[output_index] = Some(FeatureKey {
                symbol: *symbol,
                name: (*name).to_string(),
            });
        }

        Ok(Self::from_builtin_entries(
            cells,
            entries,
            specs.len(),
            names,
        ))
    }

    pub(crate) fn from_builtin_entries(
        cells: V,
        entries: [MaybeUninit<BuiltinFeatureEntry<F>>; M],
        feature_count: usize,
        names: Box<[Option<FeatureKey>]>,
    ) -> Self {
        debug_assert!(feature_count <= M);
        debug_assert!(names.len() <= cells.len());

        // Count features per kind, then turn the counts into contiguous
        // `(start, len)` group ranges via a running offset.
        let mut groups = [(0usize, 0usize); EVENT_KIND_COUNT];
        for slot in entries.iter().take(feature_count) {
            let entry = unsafe { slot.assume_init_ref() };
            groups[entry.kind as usize].1 += 1;
        }
        let mut offset = 0;
        for group in groups.iter_mut() {
            group.0 = offset;
            offset += group.1;
        }

        let mut features = [const { MaybeUninit::uninit() }; M];
        let mut next = groups.map(|(start, _)| start);

        for slot in entries.into_iter().take(feature_count) {
            let entry = unsafe { slot.assume_init() };
            let kind = entry.kind as usize;
            let pos = next[kind];
            next[kind] += 1;
            features[pos].write(entry.feature);
        }

        Self {
            feature_vector: cells,
            features,
            feature_count,
            groups,
            names,
            _marker: PhantomData,
        }
    }
}

fn validate_builtin_specs(specs: &[(&str, Symbol, BuiltinSpec)]) -> Result<()> {
    for (i, (name, symbol, spec)) in specs.iter().enumerate() {
        match spec {
            BuiltinSpec::Sma { period, .. } if *period < 1 => {
                return invalid_period("SMA", *symbol, name);
            }
            BuiltinSpec::Ema { period, .. } if *period < 1 => {
                return invalid_period("EMA", *symbol, name);
            }
            BuiltinSpec::SmaTimed {
                aggregation,
                window,
            } => {
                sma::validate_timed_durations(*aggregation, *window)?;
            }
            BuiltinSpec::ObvTimed {
                aggregation,
                window,
            } => {
                obv::validate_timed_durations(*aggregation, *window)?;
            }
            _ => {}
        }

        if specs[i + 1..]
            .iter()
            .any(|(other_name, other_symbol, _)| name == other_name && symbol == other_symbol)
        {
            return Err(FimlError::InvalidArgument(format!(
                "duplicate feature key: {}",
                feature_label(*symbol, name)
            )));
        }
    }
    Ok(())
}

/// Construct a single builtin feature wired to an output cell index.
fn build_builtin<F: Float>(
    symbol: Symbol,
    spec: &BuiltinSpec,
    output_index: usize,
) -> Result<BuiltinFeature<F>> {
    match spec {
        BuiltinSpec::Sma { period, .. } => sma::build_builtin(symbol, *period, output_index),
        BuiltinSpec::Ema { period, .. } => ema::build_builtin(symbol, *period, output_index),
        BuiltinSpec::SmaTimed {
            aggregation,
            window,
        } => sma::build_timed_builtin(symbol, *aggregation, *window, output_index),
        BuiltinSpec::ObvTimed {
            aggregation,
            window,
        } => obv::build_timed_builtin(symbol, *aggregation, *window, output_index),
        BuiltinSpec::DayOfWeek => Ok(BuiltinFeature::DayOfWeek(day_of_week::DayOfWeek::new(
            output_index,
        ))),
    }
}

fn invalid_period<T>(indicator_name: &str, symbol: Symbol, name: &str) -> Result<T> {
    Err(FimlError::InvalidArgument(format!(
        "{indicator_name} period must be at least 1 for {}",
        feature_label(symbol, name)
    )))
}

fn feature_label(symbol: Symbol, name: &str) -> String {
    match resolve(symbol) {
        Some(symbol_name) => format!("{symbol_name}:{name}"),
        None => format!("{symbol:?}:{name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, symbols};

    type Fv<const N: usize, const M: usize> =
        IndicatorFeatureVector<f64, ArrayFeatureVector<f64, N>, BuiltinFeature<f64>, M>;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn sma(period: usize) -> BuiltinSpec {
        BuiltinSpec::Sma { period }
    }

    fn ema(period: usize) -> BuiltinSpec {
        BuiltinSpec::Ema { period }
    }

    fn sma_timed(aggregation_millis: u64, window_millis: u64) -> BuiltinSpec {
        BuiltinSpec::SmaTimed {
            aggregation: std::time::Duration::from_millis(aggregation_millis),
            window: std::time::Duration::from_millis(window_millis),
        }
    }

    fn obv_timed(aggregation_millis: u64, window_millis: u64) -> BuiltinSpec {
        BuiltinSpec::ObvTimed {
            aggregation: std::time::Duration::from_millis(aggregation_millis),
            window: std::time::Duration::from_millis(window_millis),
        }
    }

    #[test]
    fn values_match_per_window_averages() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_2_sec", aapl, sma(2)), ("sma_5_sec", aapl, sma(5))];
        let mut fv: Fv<2, 2> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }

        // sma_2: mean(4,5) = 4.5 ; sma_5: mean(1..5) = 3.0
        assert!(approx_eq(fv.feature_vector().values()[0], 4.5));
        assert!(approx_eq(fv.feature_vector().values()[1], 3.0));
    }

    #[test]
    fn values_match_ema_average() {
        let aapl = symbols::intern("AAPL");
        let specs = [("ema_3", aapl, ema(3))];
        let mut fv: Fv<1, 1> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        for v in [10.0, 20.0, 30.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 22.5));
        assert_eq!(fv.index_of(aapl, "ema_3"), Some(0));
    }

    #[test]
    fn values_match_timed_window_average() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_timed_2_sec", aapl, sma_timed(1_000, 2_000))];
        let mut fv: Fv<1, 1> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

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
        assert_eq!(fv.index_of(aapl, "sma_timed_2_sec"), Some(0));
    }

    #[test]
    fn values_match_timed_obv_window_sum() {
        let aapl = symbols::intern("AAPL");
        let specs = [("obv_timed_2_sec", aapl, obv_timed(1_000, 2_000))];
        let mut fv: Fv<1, 1> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        for (price, volume, timestamp) in [
            (100.0, 10.0, 0),
            (101.0, 7.0, 1_000),
            (102.0, 5.0, 2_000),
            (99.0, 2.0, 3_000),
        ] {
            fv.dispatch(&Event::trade(aapl, price, volume, timestamp));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 3.0));
        assert_eq!(fv.index_of(aapl, "obv_timed_2_sec"), Some(0));
    }

    #[test]
    fn index_of_keeps_caller_order() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_5_sec", aapl, sma(5)), ("sma_10_sec", aapl, sma(10))];
        let fv: Fv<2, 2> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        assert_eq!(fv.index_of(aapl, "sma_5_sec"), Some(0));
        assert_eq!(fv.index_of(aapl, "sma_10_sec"), Some(1));
        assert_eq!(fv.index_of(aapl, "missing"), None);
    }

    #[test]
    fn index_of_distinguishes_symbols() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let specs = [("sma_5_sec", aapl, sma(5)), ("sma_5_sec", googl, sma(5))];
        let fv: Fv<2, 2> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        assert_eq!(fv.index_of(aapl, "sma_5_sec"), Some(0));
        assert_eq!(fv.index_of(googl, "sma_5_sec"), Some(1));
    }

    #[test]
    fn dispatch_distinguishes_symbols() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let specs = [("sma_2_sec", aapl, sma(2)), ("sma_2_sec", googl, sma(2))];
        let mut fv: Fv<2, 2> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        for v in [10.0, 20.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }
        for v in [100.0, 200.0] {
            fv.dispatch(&Event::price(googl, v, 0));
        }

        assert!(approx_eq(fv.feature_vector().values()[0], 15.0));
        assert!(approx_eq(fv.feature_vector().values()[1], 150.0));
    }

    #[test]
    fn routes_each_event_to_its_own_group() {
        let aapl = symbols::intern("AAPL");
        // Interleave kinds so grouping has to reorder the stored features while
        // keeping cell order (sma_3 -> cell 0, day_of_week -> cell 1).
        let specs = [
            ("sma_3_sec", aapl, sma(3)),
            ("day_of_week", aapl, BuiltinSpec::DayOfWeek),
        ];
        let mut fv: Fv<2, 2> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        // Price events touch only the SMA; the calendar cell stays zero.
        for v in [3.0, 6.0, 9.0] {
            fv.dispatch(&Event::price(aapl, v, 0));
        }
        assert!(approx_eq(fv.feature_vector().values()[0], 6.0)); // mean(3,6,9)
        assert!(approx_eq(fv.feature_vector().values()[1], 0.0)); // untouched

        // A time event touches only the calendar feature; the SMA is unchanged.
        fv.dispatch(&Event::time(1_609_459_200)); // 2021-01-01, a Friday
        assert!(approx_eq(fv.feature_vector().values()[0], 6.0)); // unchanged
        assert!(approx_eq(fv.feature_vector().values()[1], 5.0)); // Friday

        // An event kind with no subscribers is a no-op.
        fv.dispatch(&Event::order_book(aapl, 1.0, 2.0, 0));
        assert!(approx_eq(fv.feature_vector().values()[0], 6.0));
        assert!(approx_eq(fv.feature_vector().values()[1], 5.0));
    }

    #[test]
    fn survives_being_moved_after_construction() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_2_sec", aapl, sma(2))];
        let fv: Fv<1, 1> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs).unwrap();

        // Move the vector through a binding: features write by output index, so
        // moving the aggregate cannot invalidate borrowed cell references.
        let mut moved = fv;
        for v in [10.0, 20.0] {
            moved.dispatch(&Event::price(aapl, v, 0));
        }
        assert!(approx_eq(moved.feature_vector().values()[0], 15.0));
    }

    #[test]
    fn rejects_more_features_than_capacity() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_2_sec", aapl, sma(2)), ("sma_3_sec", aapl, sma(3))];
        let built: Result<Fv<1, 1>> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs);
        assert!(built.is_err());
    }

    #[test]
    fn rejects_zero_sma_period_without_panicking() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_0_sec", aapl, sma(0))];
        let built: Result<Fv<1, 1>> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs);
        assert!(built.is_err());
    }

    #[test]
    fn rejects_zero_ema_period_without_panicking() {
        let aapl = symbols::intern("AAPL");
        let specs = [("ema_0", aapl, ema(0))];
        let built: Result<Fv<1, 1>> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs);
        assert!(built.is_err());
    }

    #[test]
    fn rejects_invalid_sma_timed_spec_without_panicking() {
        let aapl = symbols::intern("AAPL");
        let zero_aggregation = [("sma_timed_zero", aapl, sma_timed(0, 1_000))];
        let non_multiple_window = [("sma_timed_non_multiple", aapl, sma_timed(1_000, 1_500))];

        let built_zero: Result<Fv<1, 1>> = IndicatorFeatureVector::from_builtin_specs(
            ArrayFeatureVector::new(),
            &zero_aggregation,
        );
        let built_non_multiple: Result<Fv<1, 1>> = IndicatorFeatureVector::from_builtin_specs(
            ArrayFeatureVector::new(),
            &non_multiple_window,
        );

        assert!(built_zero.is_err());
        assert!(built_non_multiple.is_err());
    }

    #[test]
    fn rejects_duplicate_feature_key() {
        let aapl = symbols::intern("AAPL");
        let specs = [("sma_2_sec", aapl, sma(2)), ("sma_2_sec", aapl, sma(3))];
        let built: Result<Fv<2, 2>> =
            IndicatorFeatureVector::from_builtin_specs(ArrayFeatureVector::new(), &specs);
        assert!(built.is_err());
    }
}

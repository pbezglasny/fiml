use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::features::builtin::BuiltinFeature;
use crate::features::compiler::{Compilation, compile};
use crate::features::definition::FeatureSet;
use crate::features::event::{EVERY_EVENT_GROUP, Event, FEATURE_GROUP_COUNT};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, Result};

/// Runtime update contract implemented by each concrete feature adapter.
pub trait Feature<F: Float> {
    fn update<O: FeatureVector<F = F>>(&mut self, event: &Event<F>, output: &mut O);
}

pub trait IndicatorFeatures {
    type F: Float;
    type FeatureVector: FeatureVector<F = Self::F>;

    fn feature_vector(&self) -> &Self::FeatureVector;
    fn dispatch(&mut self, event: &Event<Self::F>) -> Result<()>;
    fn validate_dispatch(&self, event: &Event<Self::F>) -> Result<()>;
    fn index_of(&self, canonical_name: &str) -> Option<usize>;
}

/// Fixed-capacity compiled indicator storage and allocation-free dispatcher.
///
/// `V` must contain exactly one cell for every compiled output. `M` is the
/// maximum number of indicator instances, not the number of output cells.
pub struct IndicatorFeatureVector<F, V, const M: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    feature_vector: V,
    features: [MaybeUninit<BuiltinFeature<F>>; M],
    feature_count: usize,
    timed_features: [usize; M],
    timed_feature_count: usize,
    groups: [(usize, usize); FEATURE_GROUP_COUNT],
    names: Box<[String]>,
    last_timestamp: Option<i64>,
    _marker: PhantomData<F>,
}

impl<F, V, const M: usize> IndicatorFeatureVector<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    /// Compile `feature_set` into the caller-provided fixed-capacity storage.
    pub fn from_feature_set(mut cells: V, feature_set: &FeatureSet) -> Result<Self> {
        let compilation = compile(feature_set, cells.len(), M)?;
        for index in 0..cells.len() {
            cells.set_value_at(index, F::NAN);
        }
        Ok(Self::from_compilation(cells, compilation))
    }

    pub fn feature_vector(&self) -> &V {
        &self.feature_vector
    }

    /// Canonical names borrowed in output-cell order.
    pub fn feature_names(&self) -> &[String] {
        &self.names
    }

    pub fn last_timestamp(&self) -> Option<i64> {
        self.last_timestamp
    }

    fn from_compilation(cells: V, compilation: Compilation<F>) -> Self {
        let feature_count = compilation.entries.len();
        debug_assert!(feature_count <= M);
        debug_assert_eq!(compilation.names.len(), cells.len());

        let mut groups = [(0usize, 0usize); FEATURE_GROUP_COUNT];
        for entry in &compilation.entries {
            groups[entry.route.group_index()].1 += 1;
        }
        let mut offset = 0;
        for group in &mut groups {
            group.0 = offset;
            offset += group.1;
        }

        let mut features = [const { MaybeUninit::uninit() }; M];
        let mut next = groups.map(|(start, _)| start);
        for entry in compilation.entries {
            let group = entry.route.group_index();
            let position = next[group];
            next[group] += 1;
            features[position].write(entry.feature);
        }

        let mut timed_features = [0; M];
        let mut timed_feature_count = 0;
        for (index, slot) in features.iter().enumerate().take(feature_count) {
            let feature = unsafe { slot.assume_init_ref() };
            if feature.observes_time() {
                timed_features[timed_feature_count] = index;
                timed_feature_count += 1;
            }
        }

        Self {
            feature_vector: cells,
            features,
            feature_count,
            timed_features,
            timed_feature_count,
            groups,
            names: compilation.names,
            last_timestamp: None,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn run_group(&mut self, (start, len): (usize, usize), event: &Event<F>) {
        // SAFETY: compilation initializes exactly `feature_count` entries, and
        // the group ranges partition that initialized prefix.
        for slot in &mut self.features[start..start + len] {
            let feature = unsafe { slot.assume_init_mut() };
            feature.update(event, &mut self.feature_vector);
        }
    }

    #[inline]
    fn observe_timed_features(&mut self, event: &Event<F>) {
        for timed_index in 0..self.timed_feature_count {
            let feature_index = self.timed_features[timed_index];
            let feature = unsafe { self.features[feature_index].assume_init_mut() };
            feature.observe(event, &mut self.feature_vector);
        }
    }
}

impl<F, V, const M: usize> IndicatorFeatures for IndicatorFeatureVector<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    type F = F;
    type FeatureVector = V;

    fn feature_vector(&self) -> &Self::FeatureVector {
        &self.feature_vector
    }

    fn dispatch(&mut self, event: &Event<F>) -> Result<()> {
        self.validate_dispatch(event)?;
        self.run_group(self.groups[event.kind() as usize], event);
        self.run_group(self.groups[EVERY_EVENT_GROUP], event);
        self.observe_timed_features(event);
        self.last_timestamp = Some(event.timestamp());
        Ok(())
    }

    fn validate_dispatch(&self, event: &Event<F>) -> Result<()> {
        if let Some(previous_timestamp) = self.last_timestamp
            && event.timestamp() < previous_timestamp
        {
            return Err(FimlError::TimestampOutOfOrder {
                symbol: event.symbol(),
                event_kind: event.kind(),
                timestamp: event.timestamp(),
                previous_timestamp,
            });
        }
        Ok(())
    }

    fn index_of(&self, canonical_name: &str) -> Option<usize> {
        self.names.iter().position(|name| name == canonical_name)
    }
}

impl<F, V, const M: usize> Drop for IndicatorFeatureVector<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    fn drop(&mut self) {
        // SAFETY: the initialized prefix is exactly `0..feature_count`.
        for slot in &mut self.features[..self.feature_count] {
            unsafe { slot.assume_init_drop() };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::features::{IndicatorSpec, ScopedIndicator, TimeWindows, TradeSide, ValueSource};
    use crate::{ArrayFeatureVector, FeatureVector, WarmupPolicy, symbols};

    type Vector<const N: usize, const M: usize> =
        IndicatorFeatureVector<f64, ArrayFeatureVector<f64, N>, M>;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn grouped_outputs_use_canonical_group_and_authored_window_order() {
        let feature_set = FeatureSet::new(vec![
            ScopedIndicator::symbol(
                "AAPL",
                IndicatorSpec::Sma {
                    source: ValueSource::Price,
                    windows: vec![5, 2],
                    warmup_policy: WarmupPolicy::FullWindow,
                },
            ),
            ScopedIndicator::global(IndicatorSpec::DayOfWeek),
        ]);
        let mut vector: Vector<3, 2> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let aapl = symbols::intern("AAPL");

        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            vector
                .dispatch(&Event::price(aapl, value, 1_609_459_200_000))
                .unwrap();
        }

        assert_eq!(
            vector.feature_names(),
            ["clock:day_of_week", "aapl:price:sma:5", "aapl:price:sma:2"]
        );
        assert!(approx_eq(vector.feature_vector().values()[0], 5.0));
        assert!(approx_eq(vector.feature_vector().values()[1], 3.0));
        assert!(approx_eq(vector.feature_vector().values()[2], 4.5));
        assert_eq!(vector.index_of("aapl:price:sma:2"), Some(2));
    }

    #[test]
    fn cvd_builder_dispatches_grouped_trade_side_outputs() {
        let feature_set = FeatureSet::builder().cvd("AAPL", [2, 3]).build();
        let mut vector: Vector<2, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let aapl = symbols::intern("AAPL");

        vector
            .dispatch(&Event::trade(
                aapl,
                100.0,
                10.0,
                0,
                Some(TradeSide::AgressorBuy),
            ))
            .unwrap();
        vector
            .dispatch(&Event::trade(
                aapl,
                99.0,
                3.0,
                1,
                Some(TradeSide::AgressorSell),
            ))
            .unwrap();
        vector
            .dispatch(&Event::trade(
                aapl,
                101.0,
                7.0,
                2,
                Some(TradeSide::AgressorBuy),
            ))
            .unwrap();

        assert_eq!(
            vector.feature_names(),
            ["aapl:trade:cvd:2", "aapl:trade:cvd:3"]
        );
        assert_eq!(vector.feature_vector().values(), [4.0, 14.0]);
    }

    #[test]
    fn timed_group_uses_one_runtime_indicator() {
        let feature_set = FeatureSet::new(vec![ScopedIndicator::symbol(
            "AAPL",
            IndicatorSpec::SmaTimed {
                source: ValueSource::Price,
                time_windows: TimeWindows::new(
                    Duration::from_secs(1),
                    vec![Duration::from_secs(2), Duration::from_secs(3)],
                ),
                warmup_policy: WarmupPolicy::FullWindow,
            },
        )]);
        let vector: Vector<2, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();

        assert_eq!(vector.feature_count, 1);
        assert_eq!(vector.feature_names().len(), 2);
    }

    #[test]
    fn timed_features_warm_and_expire_on_every_event() {
        let feature_set = FeatureSet::builder()
            .trade_count_timed("AAPL", Duration::from_secs(1), Duration::from_secs(2))
            .build();
        let mut vector: Vector<1, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");

        assert!(vector.feature_vector().values()[0].is_nan());
        vector
            .dispatch(&Event::trade(aapl, 100.0, 1.0, 0, None))
            .unwrap();
        vector
            .dispatch(&Event::trade(aapl, 101.0, 1.0, 1_000, None))
            .unwrap();
        assert!(vector.feature_vector().values()[0].is_nan());

        vector.dispatch(&Event::time(2_000)).unwrap();
        assert_eq!(vector.feature_vector().values(), [1.0]);

        vector
            .dispatch(&Event::trade(googl, 50.0, 1.0, 3_000, None))
            .unwrap();
        assert_eq!(vector.feature_vector().values(), [0.0]);
    }

    #[test]
    fn timed_sma_reverts_to_nan_when_its_ready_window_becomes_empty() {
        let feature_set = FeatureSet::builder()
            .sma_timed_with_warmup(
                "AAPL",
                Duration::from_secs(1),
                [Duration::from_secs(2)],
                WarmupPolicy::FirstValue,
            )
            .build();
        let mut vector: Vector<1, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let aapl = symbols::intern("AAPL");

        vector.dispatch(&Event::price(aapl, 10.0, 0)).unwrap();
        assert_eq!(vector.feature_vector().values(), [10.0]);

        vector.dispatch(&Event::time(2_000)).unwrap();
        assert!(vector.feature_vector().values()[0].is_nan());
    }

    #[test]
    fn output_storage_must_match_exactly() {
        let feature_set = FeatureSet::new(vec![ScopedIndicator::global(IndicatorSpec::DayOfWeek)]);
        let result: Result<Vector<2, 1>> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set);

        assert!(matches!(
            result,
            Err(FimlError::OutputCountMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn global_watermark_covers_unconsumed_events() {
        let feature_set = FeatureSet::new(vec![ScopedIndicator::symbol(
            "AAPL",
            IndicatorSpec::Sma {
                source: ValueSource::Price,
                windows: vec![2],
                warmup_policy: WarmupPolicy::FullWindow,
            },
        )]);
        let mut vector: Vector<1, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let googl = symbols::intern("GOOGL");

        vector.dispatch(&Event::time(100)).unwrap();
        let error = vector.dispatch(&Event::price(googl, 10.0, 99)).unwrap_err();

        assert!(matches!(
            error,
            FimlError::TimestampOutOfOrder {
                timestamp: 99,
                previous_timestamp: 100,
                ..
            }
        ));
    }
}

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

    /// Advance a time-decaying feature's clock to `now` and rewrite its output
    /// cells, without feeding it any new data.
    ///
    /// Features whose value depends only on the data they consume keep the
    /// default no-op. Time-decaying features override it so a window that has
    /// aged out reports its decayed value instead of a frozen one, whatever
    /// symbol or kind the dispatched event carried.
    fn advance_to<O: FeatureVector<F = F>>(&mut self, _now: i64, _output: &mut O) {}
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
    groups: [(usize, usize); FEATURE_GROUP_COUNT],
    /// Positions in `features` of the time-decaying features, advanced on every
    /// dispatch whatever the event's kind or symbol.
    time_decaying: Box<[usize]>,
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
    pub fn from_feature_set(cells: V, feature_set: &FeatureSet) -> Result<Self> {
        let compilation = compile(feature_set, cells.len(), M)?;
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
        let mut time_decaying = Vec::new();
        for entry in compilation.entries {
            let group = entry.route.group_index();
            let position = next[group];
            next[group] += 1;
            if entry.time_decaying {
                time_decaying.push(position);
            }
            features[position].write(entry.feature);
        }
        // Grouping reorders features, so collect the positions after placement
        // and walk them in storage order.
        time_decaying.sort_unstable();

        Self {
            feature_vector: cells,
            features,
            feature_count,
            groups,
            time_decaying: time_decaying.into_boxed_slice(),
            names: compilation.names,
            last_timestamp: None,
            _marker: PhantomData,
        }
    }

    /// Age every time-decaying window to `now` before any new data is applied,
    /// so a window that has emptied reports its decayed value even when no
    /// event for its symbol has arrived.
    #[inline]
    fn advance_time_decaying(&mut self, now: i64) {
        for cursor in 0..self.time_decaying.len() {
            let position = self.time_decaying[cursor];
            // SAFETY: the recorded positions are a subset of the initialized
            // prefix written during compilation.
            let feature = unsafe { self.features[position].assume_init_mut() };
            feature.advance_to(now, &mut self.feature_vector);
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
        // Ordering matters: validation first so a rejected event leaves no
        // trace, then decay, so expiry precedes insertion exactly once per
        // event and the groups below observe an already-aged window.
        self.advance_time_decaying(event.timestamp());
        self.run_group(self.groups[event.kind() as usize], event);
        self.run_group(self.groups[EVERY_EVENT_GROUP], event);
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
    use crate::features::{IndicatorDef, IndicatorSpec, TimeWindows, TradeSide, ValueSource};
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    type Vector<const N: usize, const M: usize> =
        IndicatorFeatureVector<f64, ArrayFeatureVector<f64, N>, M>;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn grouped_outputs_keep_definition_and_window_order() {
        let feature_set = FeatureSet::new(vec![
            IndicatorDef::symbol(
                "AAPL",
                IndicatorSpec::Sma {
                    source: ValueSource::Price,
                    windows: vec![5, 2],
                },
            ),
            IndicatorDef::global(IndicatorSpec::DayOfWeek),
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
            ["AAPL:price:sma:5", "AAPL:price:sma:2", "clock:day_of_week"]
        );
        assert!(approx_eq(vector.feature_vector().values()[0], 3.0));
        assert!(approx_eq(vector.feature_vector().values()[1], 4.5));
        assert!(approx_eq(vector.feature_vector().values()[2], 5.0));
        assert_eq!(vector.index_of("AAPL:price:sma:2"), Some(1));
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
            ["AAPL:trade:cvd:2", "AAPL:trade:cvd:3"]
        );
        assert_eq!(vector.feature_vector().values(), [4.0, 14.0]);
    }

    #[test]
    fn timed_group_uses_one_runtime_indicator() {
        let feature_set = FeatureSet::new(vec![IndicatorDef::symbol(
            "AAPL",
            IndicatorSpec::SmaTimed {
                source: ValueSource::Price,
                time_windows: TimeWindows::new(
                    Duration::from_secs(1),
                    vec![Duration::from_secs(2), Duration::from_secs(3)],
                ),
            },
        )]);
        let vector: Vector<2, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();

        assert_eq!(vector.feature_count, 1);
        assert_eq!(vector.feature_names().len(), 2);
    }

    #[test]
    fn output_storage_must_match_exactly() {
        let feature_set = FeatureSet::new(vec![IndicatorDef::global(IndicatorSpec::DayOfWeek)]);
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

    fn trade_count_set(symbol: &str, aggregation: Duration, window: Duration) -> IndicatorDef {
        IndicatorDef::symbol(
            symbol,
            IndicatorSpec::TradeCountTimed {
                aggregation,
                window,
            },
        )
    }

    #[test]
    fn time_decaying_window_ages_without_an_event_for_its_symbol() {
        let feature_set = FeatureSet::new(vec![trade_count_set(
            "AAPL",
            Duration::from_secs(1),
            Duration::from_secs(2),
        )]);
        let mut vector: Vector<1, 1> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");

        for timestamp in [0, 100, 200] {
            vector
                .dispatch(&Event::trade(aapl, 100.0, 1.0, timestamp, None))
                .unwrap();
        }
        assert_eq!(vector.feature_vector().values(), [3.0]);

        // A clock tick carrying no market data still ages the window.
        vector.dispatch(&Event::time(3_600_000)).unwrap();
        assert_eq!(vector.feature_vector().values(), [0.0]);

        // So does a trade for an entirely different symbol.
        vector
            .dispatch(&Event::trade(aapl, 100.0, 1.0, 3_600_001, None))
            .unwrap();
        assert_eq!(vector.feature_vector().values(), [1.0]);
        vector
            .dispatch(&Event::trade(googl, 10.0, 1.0, 7_200_000, None))
            .unwrap();
        assert_eq!(vector.feature_vector().values(), [0.0]);
    }

    #[test]
    fn decay_leaves_a_symbol_alone_until_its_first_event() {
        let feature_set = FeatureSet::new(vec![
            trade_count_set("AAPL", Duration::from_secs(1), Duration::from_secs(2)),
            trade_count_set("GOOGL", Duration::from_secs(1), Duration::from_secs(2)),
        ]);
        let mut vector: Vector<2, 2> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        // Cells start at NaN the way a compiled extractor prefills them.
        vector.feature_vector.set_value_at(0, f64::NAN);
        vector.feature_vector.set_value_at(1, f64::NAN);
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");

        vector
            .dispatch(&Event::trade(aapl, 100.0, 1.0, 0, None))
            .unwrap();

        // "No GOOGL trades in the last 2s" is not a claim we can make over a
        // window nobody has observed yet, so decay must not turn the warm-up
        // cell into a zero.
        assert_eq!(vector.feature_vector().values()[0], 1.0);
        assert!(vector.feature_vector().values()[1].is_nan());

        // Once GOOGL has traded, its window decays like any other.
        vector
            .dispatch(&Event::trade(googl, 10.0, 1.0, 1, None))
            .unwrap();
        assert_eq!(vector.feature_vector().values()[1], 1.0);
        vector.dispatch(&Event::time(3_600_000)).unwrap();
        assert_eq!(vector.feature_vector().values(), [0.0, 0.0]);
    }

    #[test]
    fn emptied_windows_report_zero_for_sums_and_nan_for_means() {
        let time_windows = TimeWindows::new(Duration::from_secs(1), vec![Duration::from_secs(2)]);
        let feature_set = FeatureSet::new(vec![
            IndicatorDef::symbol(
                "AAPL",
                IndicatorSpec::SmaTimed {
                    source: ValueSource::TradePrice,
                    time_windows: time_windows.clone(),
                },
            ),
            IndicatorDef::symbol("AAPL", IndicatorSpec::ObvTimed { time_windows }),
            trade_count_set("AAPL", Duration::from_secs(1), Duration::from_secs(2)),
        ]);
        let mut vector: Vector<3, 3> =
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap();
        let aapl = symbols::intern("AAPL");

        vector
            .dispatch(&Event::trade(aapl, 100.0, 5.0, 0, None))
            .unwrap();
        vector
            .dispatch(&Event::trade(aapl, 110.0, 5.0, 1_000, None))
            .unwrap();
        assert!(approx_eq(vector.feature_vector().values()[0], 105.0));

        vector.dispatch(&Event::time(3_600_000)).unwrap();

        let values = vector.feature_vector().values();
        // The mean of no samples is undefined; the sums are genuinely zero.
        assert!(
            values[0].is_nan(),
            "timed SMA should be NaN, got {}",
            values[0]
        );
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
    }

    #[test]
    fn interleaved_time_events_do_not_change_values_at_trade_rows() {
        let time_windows =
            TimeWindows::new(Duration::from_millis(100), vec![Duration::from_millis(500)]);
        let feature_set = FeatureSet::new(vec![
            IndicatorDef::symbol(
                "AAPL",
                IndicatorSpec::SmaTimed {
                    source: ValueSource::TradePrice,
                    time_windows: time_windows.clone(),
                },
            ),
            IndicatorDef::symbol("AAPL", IndicatorSpec::ObvTimed { time_windows }),
            trade_count_set(
                "AAPL",
                Duration::from_millis(100),
                Duration::from_millis(500),
            ),
        ]);
        let build = || -> Vector<3, 3> {
            IndicatorFeatureVector::from_feature_set(ArrayFeatureVector::new(), &feature_set)
                .unwrap()
        };
        let mut plain = build();
        let mut with_heartbeats = build();
        let aapl = symbols::intern("AAPL");

        // Decay is monotone and idempotent in the as-of timestamp, so extra
        // clock ticks between trades add rows to a stream without changing the
        // values observed at the trades themselves. This is what keeps a live
        // Rust stream carrying heartbeats in parity with a heartbeat-free
        // Python replay of the same trades.
        for step in 0..40i64 {
            let timestamp = step * 137;
            let price = 100.0 + (step % 5) as f64;
            plain
                .dispatch(&Event::trade(aapl, price, 1.0, timestamp, None))
                .unwrap();

            with_heartbeats.dispatch(&Event::time(timestamp)).unwrap();
            with_heartbeats.dispatch(&Event::time(timestamp)).unwrap();
            with_heartbeats
                .dispatch(&Event::trade(aapl, price, 1.0, timestamp, None))
                .unwrap();

            assert_eq!(
                plain.feature_vector().values(),
                with_heartbeats.feature_vector().values(),
                "heartbeats changed the value at step {step}"
            );
        }
    }

    #[test]
    fn global_watermark_covers_unconsumed_events() {
        let feature_set = FeatureSet::new(vec![IndicatorDef::symbol(
            "AAPL",
            IndicatorSpec::Sma {
                source: ValueSource::Price,
                windows: vec![2],
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

//! As-of lags sharing finite bucket-last observations across elapsed-time windows.

use std::time::Duration;

use super::{Evaluation, change::ChangeKind};
use crate::{FeatureVector, HeapRingBuffer, InvalidTransformationDefinitionError, RingBuffer};

const MAX_HISTORY_SLOTS: u128 = 1_000_000;

pub(super) fn validate_window(
    window: Duration,
    aggregation: Duration,
) -> Result<(), InvalidTransformationDefinitionError> {
    use InvalidTransformationDefinitionError as Error;
    if window.is_zero() {
        return Err(Error::LagWindowZero);
    }
    if aggregation.is_zero() {
        return Err(Error::LagAggregationZero);
    }
    for duration in [window, aggregation] {
        if !duration.subsec_nanos().is_multiple_of(1_000_000) {
            return Err(Error::LagDurationPrecision);
        }
        if duration.as_millis() > i64::MAX as u128 {
            return Err(Error::LagDurationOutOfRange);
        }
    }
    if window < aggregation || !window.as_millis().is_multiple_of(aggregation.as_millis()) {
        return Err(Error::LagWindowNotMultipleOfAggregation);
    }
    if window.as_millis() / aggregation.as_millis() + 2 > MAX_HISTORY_SLOTS {
        return Err(Error::TimedHistoryTooLarge);
    }
    Ok(())
}

/// One input's finite bucket-last observations, bounded by its largest lag and aggregation.
pub(crate) struct TimedTransformer {
    input: usize,
    // Optional change formula, elapsed milliseconds, and authored output index.
    outputs: Box<[(Option<ChangeKind>, i64, usize)]>,
    visible: Box<[f64]>,
    history: HeapRingBuffer<(i64, f64)>,
    max_lag: i64,
    aggregation: i64,
    clock: Option<i64>,
    current: f64,
}

impl TimedTransformer {
    pub(super) fn new(
        input: usize,
        aggregation: i64,
        outputs: Vec<(Option<ChangeKind>, i64, usize)>,
    ) -> Self {
        let max_lag = outputs.iter().map(|(_, lag, _)| *lag).max().unwrap();
        Self {
            input,
            visible: vec![f64::NAN; outputs.len()].into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            // Keep the boundary bucket, the current partial bucket, and an older predecessor.
            history: HeapRingBuffer::new((max_lag / aggregation) as usize + 2),
            max_lag,
            aggregation,
            clock: None,
            current: f64::NAN,
        }
    }

    pub(super) fn apply<V: FeatureVector>(
        &mut self,
        input: &[f64],
        observed: &[bool],
        output: &mut V,
        output_observed: &mut [bool],
        evaluation: Evaluation<'_>,
    ) {
        let now = evaluation.timestamps[self.input];
        let refreshed =
            evaluation.advance && now.is_some() && (observed[self.input] || now > self.clock);
        if refreshed {
            let now = now.unwrap();
            self.clock = Some(now);
            // Preserve the newest predecessor: sparse histories still answer arbitrarily old as-of values.
            if let Some(cutoff) = now.checked_sub(self.max_lag) {
                while self
                    .history
                    .peek_front_at(1)
                    .is_some_and(|&(time, _)| time <= cutoff)
                {
                    self.history.pop_front();
                }
            }
            if observed[self.input] {
                self.current = input[self.input];
                if self.current.is_finite() {
                    if self.history.peek_back().is_some_and(|&(time, _)| {
                        time.div_euclid(self.aggregation) == now.div_euclid(self.aggregation)
                    }) {
                        self.history.pop_back();
                    }
                    self.history.push_back((now, self.current));
                }
            }
            for ((kind, lag, _), visible) in self.outputs.iter().zip(&mut self.visible) {
                *visible = if self.current.is_finite() {
                    now.checked_sub(*lag)
                        .and_then(|cutoff| as_of(&self.history, cutoff))
                        .map_or(f64::NAN, |previous| {
                            kind.as_ref()
                                .map_or(previous, |kind| kind.calculate(self.current, previous))
                        })
                } else {
                    f64::NAN
                };
            }
        }
        for ((_, _, index), &value) in self.outputs.iter().zip(&self.visible) {
            output.set_value_at(*index, value);
            output_observed[*index] = refreshed;
        }
    }
}

fn as_of(history: &HeapRingBuffer<(i64, f64)>, cutoff: i64) -> Option<f64> {
    let (mut lower, mut upper) = (0, history.len());
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if history.peek_front_at(middle).unwrap().0 <= cutoff {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    lower
        .checked_sub(1)
        .and_then(|index| history.peek_front_at(index))
        .map(|&(_, value)| value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::transformers::{Transformer, compile};
    use crate::{ArrayFeatureVector, FeatureId, TransformerDefinition as T};

    fn equal(actual: &[f64], expected: &[f64]) {
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                actual == expected
                    || actual.is_nan() && expected.is_nan()
                    || (actual - expected).abs() < 1e-12,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn shares_history_and_matches_unbounded_as_of_oracle_through_wraparound() {
        let id = FeatureId::new;
        let mut operations = compile(
            &[
                T::delta_timed(id("x"), id("d"), Duration::from_millis(7)),
                T::lagged_timed(id("x"), id("l"), Duration::from_millis(1)),
                T::simple_return_timed(id("x"), id("r"), Duration::from_millis(3)),
                T::log_return_timed(id("x"), id("log"), Duration::from_millis(7)),
                T::lagged_timed(id("x"), id("copy"), Duration::from_millis(7)),
            ],
            &[id("x")],
        );
        assert_eq!(operations.len(), 1);
        let Transformer::Timed(group) = &mut operations[0] else {
            panic!("expected shared timed group")
        };
        assert_eq!(group.history.capacity(), 9);
        let mut output = ArrayFeatureVector::<5>::new();
        let mut mask = [false; 5];
        let mut history = Vec::<(i64, f64)>::new();
        let mut now = -100_i64;
        let mut current = f64::NAN;
        let mut previous_time = None;
        for i in 0..1000 {
            // Dense milliseconds, same-time replacements, long gaps and missing observations.
            now += if i % 41 == 0 { 100 } else { (i % 3) as i64 };
            let observed = i % 5 != 0;
            let value = if i % 17 == 0 {
                f64::INFINITY
            } else {
                (i % 11) as f64 - 4.
            };
            if observed {
                current = value;
                if value.is_finite() {
                    history.push((now, value));
                }
            }
            let evaluation = Evaluation {
                timestamps: &[Some(now)],
                global_timestamp: Some(now),
                advance: true,
            };
            group.apply(&[value], &[observed], &mut output, &mut mask, evaluation);
            let lookup = |lag| {
                history
                    .iter()
                    .rev()
                    .find(|&&(time, _)| time <= now - lag)
                    .map(|&(_, value)| value)
            };
            let mut expected = [f64::NAN; 5];
            if current.is_finite() {
                expected[0] = lookup(7).map_or(f64::NAN, |p| current - p);
                expected[1] = lookup(1).unwrap_or(f64::NAN);
                expected[2] =
                    lookup(3).map_or(f64::NAN, |p| ChangeKind::SimpleReturn.calculate(current, p));
                expected[3] =
                    lookup(7).map_or(f64::NAN, |p| ChangeKind::LogReturn.calculate(current, p));
                expected[4] = lookup(7).unwrap_or(f64::NAN);
            }
            equal(output.values(), &expected);
            assert_eq!(mask, [observed || previous_time != Some(now); 5]);
            assert!(group.history.len() <= 8);
            previous_time = Some(now);
        }
    }

    #[test]
    fn timestamp_extremes_and_missing_values_preserve_exact_history() {
        let mut group = TimedTransformer::new(
            0,
            1,
            vec![(None, 1, 0), (Some(ChangeKind::LogReturn), 1, 1)],
        );
        let mut output = ArrayFeatureVector::<2>::new();
        let mut mask = [false; 2];
        for (now, value, observed, expected) in [
            (i64::MIN, 1e-300, true, [f64::NAN; 2]),
            (i64::MIN + 1, 1e300, true, [1e-300, 1381.5510557964274]),
            (0, 99., false, [1e300, 0.]),
            (1, f64::NAN, true, [f64::NAN; 2]),
            (2, 99., false, [f64::NAN; 2]),
            (i64::MAX, 1e-300, true, [1e300, -1381.5510557964274]),
        ] {
            group.apply(
                &[value],
                &[observed],
                &mut output,
                &mut mask,
                Evaluation {
                    timestamps: &[Some(now)],
                    global_timestamp: Some(now),
                    advance: true,
                },
            );
            equal(output.values(), &expected);
        }
        let history_len = group.history.len();
        group.apply(
            &[99.],
            &[true],
            &mut output,
            &mut mask,
            Evaluation {
                timestamps: &[Some(i64::MAX)],
                global_timestamp: Some(i64::MAX),
                advance: false,
            },
        );
        equal(output.values(), &[1e300, -1381.5510557964274]);
        assert_eq!(mask, [false; 2]);
        assert_eq!(group.history.len(), history_len);
    }

    #[test]
    fn rejects_invalid_precision_zero_and_unrepresentable_storage() {
        for window in [
            Duration::ZERO,
            Duration::from_nanos(1),
            Duration::from_micros(1001),
            Duration::MAX,
            Duration::from_millis(i64::MAX as u64),
        ] {
            assert!(validate_window(window, Duration::from_millis(1)).is_err());
        }
        for window in [
            Duration::from_millis(1),
            Duration::from_secs(1),
            Duration::from_millis(999_998),
        ] {
            assert!(validate_window(window, Duration::from_millis(1)).is_ok());
        }
    }

    #[test]
    fn aggregation_shares_only_compatible_histories_and_bounds_long_windows() {
        let id = FeatureId::new;
        let day = Duration::from_secs(86_400);
        let second = Duration::from_secs(1);
        let minute = Duration::from_secs(60);
        let operations = compile(
            &[
                T::lagged_timed_with_aggregation(id("x"), id("lag"), day, second),
                T::delta_timed_with_aggregation(id("x"), id("d"), day, second),
                T::simple_return_timed_with_aggregation(id("x"), id("r"), day, second),
                T::log_return_timed_with_aggregation(id("x"), id("log"), day, second),
                T::lagged_timed_with_aggregation(id("x"), id("coarse"), day, minute),
            ],
            &[id("x")],
        );
        assert_eq!(operations.len(), 2);
        for operation in operations {
            let Transformer::Timed(group) = operation else {
                panic!("expected timed group")
            };
            if group.aggregation == 1000 {
                assert_eq!(group.outputs.len(), 4);
                assert_eq!(group.history.capacity(), 86_402);
            } else {
                assert_eq!(group.outputs.len(), 1);
                assert_eq!(group.history.capacity(), 1_442);
            }
        }
        use InvalidTransformationDefinitionError as Error;
        for (window, aggregation, expected) in [
            (day, Duration::from_millis(1), Error::TimedHistoryTooLarge),
            (
                Duration::from_secs(3_600_000_000),
                second,
                Error::TimedHistoryTooLarge,
            ),
            (second, Duration::ZERO, Error::LagAggregationZero),
            (second, Duration::from_nanos(1), Error::LagDurationPrecision),
            (
                second,
                Duration::from_secs(u64::MAX),
                Error::LagDurationOutOfRange,
            ),
            (second, minute, Error::LagWindowNotMultipleOfAggregation),
            (
                second,
                Duration::from_millis(300),
                Error::LagWindowNotMultipleOfAggregation,
            ),
            (
                Duration::from_millis(999_999),
                Duration::from_millis(1),
                Error::TimedHistoryTooLarge,
            ),
        ] {
            assert_eq!(validate_window(window, aggregation), Err(expected));
        }
        assert!(validate_window(Duration::from_millis(999_998), Duration::from_millis(1)).is_ok());
        assert!(validate_window(day, second).is_ok());
    }

    #[test]
    fn coarse_history_matches_bucket_last_oracle_without_future_samples() {
        use std::collections::BTreeMap;
        for aggregation in [1, 2, 10, 1000] {
            let lag = 3 * aggregation;
            let mut group =
                TimedTransformer::new(0, aggregation, vec![(None, lag, 0), (None, aggregation, 1)]);
            let mut history = BTreeMap::<i64, (i64, f64)>::new();
            let mut output = ArrayFeatureVector::<2>::new();
            let mut observed = [false; 2];
            let mut now = -10 * aggregation;
            let mut current = f64::NAN;
            for i in 0..2000 {
                now += if i % 41 == 0 {
                    10 * aggregation
                } else {
                    i % (aggregation + 1)
                };
                let input_observed = i % 5 != 0;
                let value = if i % 17 == 0 { f64::NAN } else { i as f64 };
                if input_observed {
                    current = value;
                    if value.is_finite() {
                        history.insert(now.div_euclid(aggregation), (now, value));
                    }
                }
                group.apply(
                    &[value],
                    &[input_observed],
                    &mut output,
                    &mut observed,
                    Evaluation {
                        timestamps: &[Some(now)],
                        global_timestamp: Some(now),
                        advance: true,
                    },
                );
                let expected = [lag, aggregation].map(|lag| {
                    if !current.is_finite() {
                        return f64::NAN;
                    }
                    history
                        .values()
                        .rev()
                        .find(|&&(time, _)| time <= now - lag)
                        .map_or(f64::NAN, |&(_, value)| value)
                });
                equal(output.values(), &expected);
                assert!(group.history.len() <= 5);
            }
        }
    }

    #[test]
    fn coarse_boundary_retains_predecessor_and_handles_extreme_timestamps() {
        for base in [0, -30, i64::MIN, i64::MAX - 40] {
            let mut group = TimedTransformer::new(0, 10, vec![(None, 20, 0)]);
            let mut output = ArrayFeatureVector::<1>::new();
            let mut observed = [false];
            for (offset, value) in [(0, 2.), (19, 4.), (29, 6.), (30, 8.)] {
                let now = base + offset;
                group.apply(
                    &[value],
                    &[true],
                    &mut output,
                    &mut observed,
                    Evaluation {
                        timestamps: &[Some(now)],
                        global_timestamp: Some(now),
                        advance: true,
                    },
                );
            }
            equal(output.values(), &[2.]);
        }
    }
}

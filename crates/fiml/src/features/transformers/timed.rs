//! Exact as-of lags sharing finite timestamp/value history across elapsed-time windows.

use std::time::Duration;

use super::{Evaluation, change::ChangeKind};
use crate::{FeatureVector, HeapRingBuffer, InvalidTransformationDefinitionError, RingBuffer};

pub(super) fn validate_window(
    window: Duration,
) -> Result<(), InvalidTransformationDefinitionError> {
    use InvalidTransformationDefinitionError as Error;
    if window.is_zero() {
        return Err(Error::LagWindowZero);
    }
    if !window.subsec_nanos().is_multiple_of(1_000_000) {
        return Err(Error::LagDurationPrecision);
    }
    let millis = window.as_millis();
    if millis > i64::MAX as u128
        || millis + 1 > (isize::MAX as usize / size_of::<(i64, f64)>()) as u128
    {
        return Err(Error::LagDurationOutOfRange);
    }
    Ok(())
}

/// One input's finite as-of observations, bounded by its largest millisecond lag.
pub(crate) struct TimedTransformer {
    input: usize,
    // Optional change formula, elapsed milliseconds, and authored output index.
    outputs: Box<[(Option<ChangeKind>, i64, usize)]>,
    visible: Box<[f64]>,
    history: HeapRingBuffer<(i64, f64)>,
    max_lag: i64,
    clock: Option<i64>,
    current: f64,
}

impl TimedTransformer {
    pub(super) fn new(input: usize, outputs: Vec<(Option<ChangeKind>, i64, usize)>) -> Self {
        let max_lag = outputs.iter().map(|(_, lag, _)| *lag).max().unwrap();
        Self {
            input,
            visible: vec![f64::NAN; outputs.len()].into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            history: HeapRingBuffer::new(max_lag as usize + 1),
            max_lag,
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
                    if self
                        .history
                        .peek_back()
                        .is_some_and(|&(time, _)| time == now)
                    {
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
        assert_eq!(group.history.capacity(), 8);
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
        let mut group =
            TimedTransformer::new(0, vec![(None, 1, 0), (Some(ChangeKind::LogReturn), 1, 1)]);
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
            assert!(validate_window(window).is_err());
        }
        for window in [
            Duration::from_millis(1),
            Duration::from_secs(1),
            Duration::from_secs(3600),
        ] {
            assert!(validate_window(window).is_ok());
        }
    }
}

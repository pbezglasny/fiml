//! Constant-space streaks and ages of signed signals from any preceding scalar stage.

use super::Evaluation;
use crate::FeatureVector;

const MAX_COUNT: f64 = 9_007_199_254_740_992.; // All integers through 2^53 are exact in f64.

/// Selects a sign predicate and whether to measure persistence, observations, or time.
#[derive(Clone, Copy)]
pub(super) enum StateKind {
    PositiveStreak,
    NegativeStreak,
    ObservationsSincePositive,
    ObservationsSinceNegative,
    ObservationsSinceNonzero,
    TimeSincePositive,
    TimeSinceNegative,
    TimeSinceNonzero,
}

impl StateKind {
    pub(super) fn is_timed(self) -> bool {
        matches!(
            self,
            Self::TimeSincePositive | Self::TimeSinceNegative | Self::TimeSinceNonzero
        )
    }

    fn is_streak(self) -> bool {
        matches!(self, Self::PositiveStreak | Self::NegativeStreak)
    }

    fn matches(self, value: f64, epsilon: f64) -> bool {
        match self {
            Self::PositiveStreak | Self::ObservationsSincePositive | Self::TimeSincePositive => {
                value > epsilon
            }
            Self::NegativeStreak | Self::ObservationsSinceNegative | Self::TimeSinceNegative => {
                value < -epsilon
            }
            Self::ObservationsSinceNonzero | Self::TimeSinceNonzero => value.abs() > epsilon,
        }
    }
}

/// Retains one signal's count or last matching clock without storing its history.
pub(crate) struct StateTransformer {
    input: usize,
    output: usize,
    kind: StateKind,
    epsilon: f64,
    count: f64,
    last_match: Option<i64>,
    clock: Option<i64>,
    finite: bool,
    visible: f64,
}

impl StateTransformer {
    pub(super) fn new(input: usize, output: usize, kind: StateKind, epsilon: f64) -> Self {
        Self {
            input,
            output,
            kind,
            epsilon,
            count: if kind.is_streak() { 0. } else { f64::NAN },
            last_match: None,
            clock: None,
            finite: false,
            visible: f64::NAN,
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
        let observed = observed[self.input];
        let value = input[self.input];
        let refreshed = if self.kind.is_timed() {
            let now = evaluation.timestamps[self.input];
            let refreshed = evaluation.advance && now.is_some() && (observed || now > self.clock);
            if refreshed {
                let now = now.unwrap();
                self.clock = Some(now);
                if observed {
                    self.finite = value.is_finite();
                    if self.finite && self.kind.matches(value, self.epsilon) {
                        self.last_match = Some(now);
                    }
                }
                self.visible = if self.finite {
                    // Widen before subtracting so the entire i64 timestamp domain is supported.
                    self.last_match
                        .map_or(f64::NAN, |last| (i128::from(now) - i128::from(last)) as f64)
                } else {
                    f64::NAN
                };
            }
            refreshed
        } else {
            if observed {
                self.visible = if value.is_finite() {
                    let matches = self.kind.matches(value, self.epsilon);
                    if matches {
                        self.count = if self.kind.is_streak() {
                            (self.count + 1.).min(MAX_COUNT)
                        } else {
                            0.
                        };
                    } else if self.kind.is_streak() {
                        self.count = 0.;
                    } else if self.count < MAX_COUNT {
                        self.count += 1.;
                    }
                    self.count
                } else {
                    f64::NAN
                };
            }
            observed
        };
        output.set_value_at(self.output, self.visible);
        output_observed[self.output] = refreshed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrayFeatureVector;

    const KINDS: [StateKind; 8] = [
        StateKind::PositiveStreak,
        StateKind::NegativeStreak,
        StateKind::ObservationsSincePositive,
        StateKind::ObservationsSinceNegative,
        StateKind::ObservationsSinceNonzero,
        StateKind::TimeSincePositive,
        StateKind::TimeSinceNegative,
        StateKind::TimeSinceNonzero,
    ];

    fn run(
        state: &mut StateTransformer,
        value: f64,
        observed: bool,
        now: Option<i64>,
        advance: bool,
    ) -> (f64, bool) {
        let mut output = ArrayFeatureVector::<1>::new();
        let mut mask = [false];
        state.apply(
            &[value],
            &[observed],
            &mut output,
            &mut mask,
            Evaluation {
                timestamps: &[now],
                global_timestamp: now,
                advance,
            },
        );
        (output.values()[0], mask[0])
    }

    fn equal(actual: f64, expected: f64) {
        assert!(
            actual == expected || actual.is_nan() && expected.is_nan(),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn literals_cover_neutral_boundaries_streak_resets_and_first_matches() {
        let nan = f64::NAN;
        let rows = [
            (0., [0., 0., nan, nan, nan, nan, nan, nan]),
            (0.5, [0., 0., nan, nan, nan, nan, nan, nan]),
            (0.5_f64.next_up(), [1., 0., 0., nan, 0., 0., nan, 0.]),
            (1., [2., 0., 0., nan, 0., 0., nan, 0.]),
            (-0.5, [0., 0., 1., nan, 1., 10., nan, 10.]),
            ((-0.5_f64).next_down(), [0., 1., 2., 0., 0., 20., 0., 0.]),
            (-2., [0., 2., 3., 0., 0., 30., 0., 0.]),
            (-0., [0., 0., 4., 1., 1., 40., 10., 10.]),
            (1., [1., 0., 0., 2., 0., 0., 20., 0.]),
        ];
        for (column, kind) in KINDS.into_iter().enumerate() {
            let mut state = StateTransformer::new(0, 0, kind, 0.5);
            for (time, (value, expected)) in rows.iter().enumerate() {
                let (actual, observed) =
                    run(&mut state, *value, true, Some(time as i64 * 10), true);
                equal(actual, expected[column]);
                assert!(observed);
            }
        }
    }

    #[test]
    fn nonfinite_inputs_preserve_state_and_clock_only_events_do_not_create_matches() {
        for kind in KINDS {
            let mut state = StateTransformer::new(0, 0, kind, 0.);
            assert!(run(&mut state, 99., false, None, true).0.is_nan());
            run(&mut state, 2., true, Some(0), true);
            let (saved, _) = run(&mut state, -2., true, Some(10), true);
            equal(run(&mut state, 99., false, Some(10), true).0, saved);
            for missing in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(run(&mut state, missing, true, Some(20), true).0.is_nan());
            }
            assert!(run(&mut state, 99., false, Some(30), true).0.is_nan());
            let (actual, _) = run(&mut state, 0., true, Some(40), true);
            let expected = match kind {
                StateKind::PositiveStreak | StateKind::NegativeStreak => 0.,
                StateKind::ObservationsSincePositive => 2.,
                StateKind::ObservationsSinceNegative | StateKind::ObservationsSinceNonzero => 1.,
                StateKind::TimeSincePositive => 40.,
                StateKind::TimeSinceNegative | StateKind::TimeSinceNonzero => 30.,
            };
            equal(actual, expected);
            let (actual, observed) = run(&mut state, 99., false, Some(50), true);
            equal(actual, expected + if kind.is_timed() { 10. } else { 0. });
            assert_eq!(observed, kind.is_timed());
        }
        let mut streak = StateTransformer::new(0, 0, StateKind::PositiveStreak, 0.);
        run(&mut streak, 1., true, None, true);
        run(&mut streak, f64::NAN, true, None, true);
        equal(run(&mut streak, 1., true, None, true).0, 2.);
    }

    #[test]
    fn counts_saturate_and_reset_while_time_subtraction_handles_i64_extremes() {
        for kind in KINDS.into_iter().filter(|kind| !kind.is_timed()) {
            let mut state = StateTransformer::new(0, 0, kind, 0.);
            state.count = MAX_COUNT - 1.;
            let value = match kind {
                StateKind::PositiveStreak => 1.,
                StateKind::NegativeStreak => -1.,
                _ => 0.,
            };
            for _ in 0..3 {
                equal(run(&mut state, value, true, None, true).0, MAX_COUNT);
            }
            let reset = match kind {
                StateKind::ObservationsSincePositive | StateKind::ObservationsSinceNonzero => 1.,
                StateKind::ObservationsSinceNegative => -1.,
                _ => 0.,
            };
            equal(run(&mut state, reset, true, None, true).0, 0.);
        }
        let mut age = StateTransformer::new(0, 0, StateKind::TimeSincePositive, 0.);
        equal(run(&mut age, 1., true, Some(i64::MIN), true).0, 0.);
        equal(
            run(&mut age, 0., true, Some(i64::MAX), true).0,
            (i128::from(i64::MAX) - i128::from(i64::MIN)) as f64,
        );
        equal(run(&mut age, 1., true, Some(i64::MAX), true).0, 0.);
    }

    #[test]
    fn timed_states_ignore_context_updates_and_clockless_inputs() {
        let mut age = StateTransformer::new(0, 0, StateKind::TimeSincePositive, 0.);
        assert!(run(&mut age, 1., true, None, false).0.is_nan());
        assert!(run(&mut age, 1., true, None, true).0.is_nan());
        equal(run(&mut age, 1., true, Some(0), true).0, 0.);
        let (actual, observed) = run(&mut age, -1., true, Some(10), false);
        equal(actual, 0.);
        assert!(!observed);
        equal(run(&mut age, -1., false, Some(10), true).0, 10.);
    }
}

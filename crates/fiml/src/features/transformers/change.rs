//! Shares finite observation history across delta and return outputs for one input.

use crate::{FeatureVector, HeapRingBuffer, RingBuffer};

/// Arithmetic applied to a current and lagged finite observation.
pub(super) enum ChangeKind {
    Delta,
    SimpleReturn,
    LogReturn,
}

impl ChangeKind {
    fn calculate(&self, current: f64, previous: f64) -> f64 {
        match self {
            Self::Delta => current - previous,
            Self::SimpleReturn if previous != 0.0 => current / previous - 1.0,
            Self::LogReturn if current > 0.0 && previous > 0.0 => current.ln() - previous.ln(),
            _ => f64::NAN,
        }
    }
}

/// One input's exact observation lags, retaining visible outputs across unrelated events.
pub(crate) struct ChangeTransformer {
    input: usize,
    // Formula, positive lag window, and authored output index.
    outputs: Box<[(ChangeKind, usize, usize)]>,
    visible: Box<[f64]>,
    history: HeapRingBuffer<f64>,
}

impl ChangeTransformer {
    pub(super) fn new(input: usize, outputs: Vec<(ChangeKind, usize, usize)>) -> Self {
        let capacity = outputs.iter().map(|(_, lag, _)| *lag).max().unwrap();
        Self {
            input,
            visible: vec![f64::NAN; outputs.len()].into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            history: HeapRingBuffer::new(capacity),
        }
    }

    pub(super) fn apply<V: FeatureVector>(
        &mut self,
        input: &[f64],
        observed: &[bool],
        output: &mut V,
        output_observed: &mut [bool],
    ) {
        if observed[self.input] {
            let current = input[self.input];
            if current.is_finite() {
                for ((kind, lag, _), visible) in self.outputs.iter().zip(&mut self.visible) {
                    *visible = self
                        .history
                        .peek_back_at(lag - 1)
                        .map_or(f64::NAN, |&previous| kind.calculate(current, previous));
                }
                self.history.push_back(current);
            } else {
                self.visible.fill(f64::NAN);
            }
        }
        for ((_, _, index), &value) in self.outputs.iter().zip(&self.visible) {
            output.set_value_at(*index, value);
            output_observed[*index] = observed[self.input];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::transformers::{Transformer, compile};
    use crate::{ArrayFeatureVector, FeatureId, TransformerDefinition as T};

    #[test]
    fn compiler_shares_one_history_across_kinds_windows_and_duplicates() {
        let id = FeatureId::new;
        let mut operations = compile(
            &[
                T::delta(id("x"), id("d3"), 3),
                T::simple_return(id("x"), id("r1"), 1),
                T::log_return(id("x"), id("l2"), 2),
                T::delta(id("x"), id("copy"), 3),
            ],
            &[id("x")],
        );
        assert_eq!(operations.len(), 1);
        let Transformer::Change(group) = &mut operations[0] else {
            panic!("expected one change group");
        };
        assert_eq!(group.history.capacity(), 3);
        assert_eq!(group.outputs.len(), 4);
        let mut output = ArrayFeatureVector::<4>::new();
        let mut observed = [false; 4];
        let missing = f64::NAN;
        let log2 = std::f64::consts::LN_2;
        for (value, expected) in [
            (2., [missing; 4]),
            (2., [missing, 0., missing, missing]),
            (4., [missing, 1., log2, missing]),
            (8., [6., 1., 2. * log2, 6.]),
            (4., [2., -0.5, 0., 2.]),
            (2., [-2., -0.5, -2. * log2, -2.]),
        ] {
            group.apply(&[value], &[true], &mut output, &mut observed);
            for (&actual, expected) in output.values().iter().zip(expected) {
                assert!(
                    (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() < 1e-12
                );
            }
            assert_eq!(observed, [true; 4]);
        }
    }

    #[test]
    fn missing_observations_preserve_history_and_reused_output_storage() {
        let mut transformer = ChangeTransformer::new(0, vec![(ChangeKind::Delta, 1, 0)]);
        let mut output = ArrayFeatureVector::<1>::new();
        let mut mask = [false];
        for (value, observed, expected, history_len) in [
            (2., true, f64::NAN, 1),
            (4., true, 2., 1),
            (f64::NAN, false, 2., 1),
            (f64::NAN, true, f64::NAN, 1),
            (f64::INFINITY, true, f64::NAN, 1),
            (f64::NEG_INFINITY, true, f64::NAN, 1),
            (99., false, f64::NAN, 1),
            (8., true, 4., 1),
            (8., true, 0., 1),
        ] {
            output.set_value_at(0, 999.);
            mask.fill(!observed);
            transformer.apply(&[value], &[observed], &mut output, &mut mask);
            let actual = output.values()[0];
            assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
            assert_eq!(mask, [observed]);
            assert_eq!(transformer.history.len(), history_len);
        }
    }

    #[test]
    fn logarithmic_returns_do_not_overflow_or_underflow_the_ratio() {
        let kind = ChangeKind::LogReturn;
        for (current, previous, expected) in [
            (1e300, 1e-300, 1381.5510557964274),
            (1e-300, 1e300, -1381.5510557964274),
        ] {
            let result = kind.calculate(current, previous);
            assert!(result.is_finite());
            assert!((result - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn formulas_cover_constant_rising_falling_and_invalid_domains() {
        use std::f64::consts::LN_2;
        for (previous, current, expected) in [
            (2., 2., [0., 0., 0.]),
            (2., 4., [2., 1., LN_2]),
            (4., 2., [-2., -0.5, -LN_2]),
            (2., 0., [-2., -1., f64::NAN]),
            (0., 2., [2., f64::NAN, f64::NAN]),
            (-0., 2., [2., f64::NAN, f64::NAN]),
            (-2., -4., [-2., 1., f64::NAN]),
            (-2., 4., [6., -3., f64::NAN]),
        ] {
            for (kind, expected) in [
                ChangeKind::Delta,
                ChangeKind::SimpleReturn,
                ChangeKind::LogReturn,
            ]
            .iter()
            .zip(expected)
            {
                let actual = kind.calculate(current, previous);
                assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
            }
        }
    }
}

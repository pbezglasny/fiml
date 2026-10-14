//! Ordered trajectory summaries sharing one finite-observation history per input.

use super::history::{History, Merge, valid_capacity};
use crate::{FeatureVector, InvalidTransformationDefinitionError, WarmupPolicy};

/// Selects an interpretable path measurement from a shared window summary.
#[derive(Clone, Copy)]
pub(super) enum TrajectoryKind {
    Slope,
    PathLength,
    PositiveMovement,
    NegativeMovement,
    DirectionChangeCount,
    MaxDrawdown,
}

/// Regression moments are relative to the first value to preserve small price changes.
#[derive(Clone, Copy, Default)]
struct Summary {
    count: usize,
    first: f64,
    last: f64,
    min: f64,
    max: f64,
    mean: f64,
    covariance: f64,
    positive: f64,
    negative: f64,
    drawdown: f64,
    changes: usize,
    first_direction: i8,
    last_direction: i8,
}

impl Summary {
    fn value(value: f64) -> Self {
        Self {
            count: 1,
            first: value,
            last: value,
            min: value,
            max: value,
            ..Self::default()
        }
    }

    fn measurement(self, kind: TrajectoryKind) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let value = match kind {
            TrajectoryKind::Slope if self.count < 2 => f64::NAN,
            TrajectoryKind::Slope => {
                let n = self.count as f64;
                self.covariance / (n * (n * n - 1.) / 12.)
            }
            TrajectoryKind::PathLength => self.positive + self.negative,
            TrajectoryKind::PositiveMovement => self.positive,
            TrajectoryKind::NegativeMovement => self.negative,
            TrajectoryKind::DirectionChangeCount => self.changes as f64,
            TrajectoryKind::MaxDrawdown => self.drawdown,
        };
        if value.is_finite() { value } else { f64::NAN }
    }
}

fn reversal(left: i8, right: i8) -> usize {
    usize::from(left != 0 && right != 0 && left != right)
}

impl Merge for Summary {
    fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }
        let count = self.count + other.count;
        let delta = other.first - self.last;
        let direction = if other.first > self.last {
            1
        } else if other.first < self.last {
            -1
        } else {
            0
        };
        let boundary_changes = if direction == 0 {
            reversal(self.last_direction, other.first_direction)
        } else {
            reversal(self.last_direction, direction) + reversal(direction, other.first_direction)
        };
        let mean_delta = (other.first - self.first) + (other.mean - self.mean);
        Self {
            count,
            first: self.first,
            last: other.last,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            mean: self.mean + mean_delta * (other.count as f64 / count as f64),
            // Segment index means differ by count / 2, cancelling the pooled divisor.
            covariance: self.covariance
                + other.covariance
                + mean_delta * (self.count as f64 * other.count as f64 / 2.),
            positive: self.positive + other.positive + delta.max(0.),
            negative: self.negative + other.negative + (-delta).max(0.),
            drawdown: self.drawdown.max(other.drawdown).max(self.max - other.min),
            changes: self.changes + other.changes + boundary_changes,
            first_direction: if self.first_direction != 0 {
                self.first_direction
            } else if direction != 0 {
                direction
            } else {
                other.first_direction
            },
            last_direction: if other.last_direction != 0 {
                other.last_direction
            } else if direction != 0 {
                direction
            } else {
                self.last_direction
            },
        }
    }
}

pub(super) fn validate_capacity(window: usize) -> Result<(), InvalidTransformationDefinitionError> {
    if window == 0 {
        return Err(InvalidTransformationDefinitionError::WindowZero);
    }
    if !valid_capacity::<Summary>(window) {
        return Err(InvalidTransformationDefinitionError::TrajectoryHistoryTooLarge);
    }
    Ok(())
}

/// Authored position and availability, retaining the visible value on unrelated events.
struct Output {
    kind: TrajectoryKind,
    warmup: WarmupPolicy,
    index: usize,
    visible: f64,
}

/// One ordered range query serves every measurement and warmup policy of this width.
struct Window {
    width: usize,
    outputs: Box<[Output]>,
}

/// Computes path measurements without copying windows or allocating during updates.
pub(crate) struct TrajectoryTransformer {
    input: usize,
    windows: Box<[Window]>,
    history: History<Summary>,
}

impl TrajectoryTransformer {
    pub(super) fn new(
        input: usize,
        outputs: Vec<(TrajectoryKind, usize, WarmupPolicy, usize)>,
    ) -> Self {
        let size = outputs.iter().map(|o| o.1).max().unwrap();
        let mut groups = std::collections::BTreeMap::<usize, Vec<Output>>::new();
        for (kind, width, warmup, index) in outputs {
            groups.entry(width).or_default().push(Output {
                kind,
                warmup,
                index,
                visible: f64::NAN,
            });
        }
        Self {
            input,
            windows: groups
                .into_iter()
                .map(|(width, outputs)| Window {
                    width,
                    outputs: outputs.into_boxed_slice(),
                })
                .collect(),
            history: History::new(size),
        }
    }

    pub(super) fn apply<V: FeatureVector>(
        &mut self,
        input: &[f64],
        observed: &[bool],
        output: &mut V,
        output_observed: &mut [bool],
    ) {
        let refreshed = observed[self.input];
        if refreshed {
            let finite = input[self.input].is_finite();
            if finite {
                self.history.push(Summary::value(input[self.input]), false);
            }
            for window in &mut self.windows {
                let summary = self.history.newest(self.history.len.min(window.width));
                for out in &mut window.outputs {
                    out.visible = if finite
                        && (self.history.len >= window.width
                            || out.warmup == WarmupPolicy::FirstValue)
                    {
                        summary.measurement(out.kind)
                    } else {
                        f64::NAN
                    };
                }
            }
        }
        for window in &self.windows {
            for out in &window.outputs {
                output.set_value_at(out.index, out.visible);
                output_observed[out.index] = refreshed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::transformers::{Transformer, compile};
    use crate::{ArrayFeatureVector, FeatureId, TransformerDefinition as T};

    const KINDS: [TrajectoryKind; 6] = [
        TrajectoryKind::Slope,
        TrajectoryKind::PathLength,
        TrajectoryKind::PositiveMovement,
        TrajectoryKind::NegativeMovement,
        TrajectoryKind::DirectionChangeCount,
        TrajectoryKind::MaxDrawdown,
    ];

    fn equal(actual: &[f64], expected: &[f64]) {
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(
                a == e || a.is_nan() && e.is_nan() || (a - e).abs() < 1e-10,
                "{actual:?} != {expected:?}"
            );
        }
    }

    fn oracle(values: &[f64]) -> [f64; 6] {
        let n = values.len() as f64;
        let mean_x = (n - 1.) / 2.;
        let origin = values[0];
        let mean_y = values.iter().map(|y| y - origin).sum::<f64>() / n;
        let numerator = values
            .iter()
            .enumerate()
            .map(|(i, y)| (i as f64 - mean_x) * ((y - origin) - mean_y))
            .sum::<f64>();
        let denominator = (0..values.len())
            .map(|i| (i as f64 - mean_x).powi(2))
            .sum::<f64>();
        let (mut path, mut positive, mut negative) = (0., 0., 0.);
        let (mut changes, mut previous) = (0., 0.);
        let (mut peak, mut drawdown) = (origin, 0_f64);
        for pair in values.windows(2) {
            let delta = pair[1] - pair[0];
            path += delta.abs();
            positive += delta.max(0.);
            negative += (-delta).max(0.);
            if delta != 0. {
                let sign = delta.signum();
                if previous != 0. && previous != sign {
                    changes += 1.;
                }
                previous = sign;
            }
            peak = peak.max(pair[1]);
            drawdown = drawdown.max(peak - pair[1]);
        }
        [
            numerator / denominator,
            path,
            positive,
            negative,
            changes,
            drawdown,
        ]
    }

    fn outputs(
        width: usize,
        warmup: WarmupPolicy,
        offset: usize,
    ) -> Vec<(TrajectoryKind, usize, WarmupPolicy, usize)> {
        KINDS
            .into_iter()
            .enumerate()
            .map(|(i, kind)| (kind, width, warmup, offset + i))
            .collect()
    }

    fn run(group: &mut TrajectoryTransformer, value: f64, observed: bool) -> Vec<f64> {
        let mut out = ArrayFeatureVector::<64>::new();
        let mut mask = [false; 64];
        group.apply(&[value], &[observed], &mut out, &mut mask);
        for window in &group.windows {
            for output in &window.outputs {
                assert_eq!(mask[output.index], observed);
            }
        }
        out.values().to_vec()
    }

    #[test]
    fn literal_paths_distinguish_shape_and_ignore_flat_deltas() {
        for (values, expected) in [
            (vec![1., 2., 3., 4.], [1., 3., 3., 0., 0., 0.]),
            (vec![1., 4., 1., 4.], [0.6, 9., 6., 3., 2., 3.]),
            (vec![4., 3., 2., 1.], [-1., 3., 0., 3., 0., 3.]),
            (vec![1., 3., 3., 1.], [0., 4., 2., 2., 1., 2.]),
            (vec![-2., -2., -2., -2.], [0.; 6]),
        ] {
            let mut group =
                TrajectoryTransformer::new(0, outputs(values.len(), WarmupPolicy::FullWindow, 0));
            let mut result = Vec::new();
            for &value in &values {
                result = run(&mut group, value, true);
            }
            equal(&result[..6], &expected);
            equal(
                &[result[1], result[2] - result[3]],
                &[result[2] + result[3], values.last().unwrap() - values[0]],
            );
        }
    }

    #[test]
    fn shared_windows_match_independent_references_through_wraparound() {
        let widths = [1, 2, 5, 21];
        for origin in [0., 1e12] {
            let mut defs = Vec::new();
            for (i, width) in widths.into_iter().enumerate() {
                defs.extend(outputs(width, WarmupPolicy::FirstValue, i * 12));
                defs.extend(outputs(width, WarmupPolicy::FullWindow, i * 12 + 6));
            }
            let mut group = TrajectoryTransformer::new(0, defs);
            let mut values = Vec::new();
            for i in 0..5000 {
                let value = origin + ((i / 3 * 17) % 23) as f64 / 1024.;
                values.push(value);
                let actual = run(&mut group, value, true);
                for (w, width) in widths.into_iter().enumerate() {
                    let expected = oracle(&values[values.len().saturating_sub(width)..]);
                    equal(&actual[w * 12..w * 12 + 6], &expected);
                    equal(
                        &actual[w * 12 + 6..w * 12 + 12],
                        &if values.len() >= width {
                            expected
                        } else {
                            [f64::NAN; 6]
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn peaks_troughs_and_reversals_leave_the_window() {
        let mut group = TrajectoryTransformer::new(0, outputs(3, WarmupPolicy::FirstValue, 0));
        for (value, drawdown, changes) in [
            (10., 0., 0.),
            (1., 9., 0.),
            (5., 9., 1.),
            (6., 0., 0.),
            (2., 4., 1.),
            (12., 4., 1.),
            (13., 0., 0.),
        ] {
            let result = run(&mut group, value, true);
            equal(&[result[5], result[4]], &[drawdown, changes]);
        }
    }

    #[test]
    fn missing_observations_and_overflow_preserve_history_and_recover() {
        let mut group = TrajectoryTransformer::new(0, outputs(2, WarmupPolicy::FirstValue, 0));
        assert!(run(&mut group, 99., false)[..6].iter().all(|v| v.is_nan()));
        equal(
            &run(&mut group, 2., true)[..6],
            &[f64::NAN, 0., 0., 0., 0., 0.],
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(run(&mut group, value, true)[..6].iter().all(|v| v.is_nan()));
        }
        let result = run(&mut group, 5., true);
        equal(&result[..6], &[3., 3., 3., 0., 0., 0.]);
        equal(&run(&mut group, 100., false)[..6], &result[..6]);
        run(&mut group, f64::MAX, true);
        let result = run(&mut group, -f64::MAX, true);
        assert!(
            result[0].is_nan() && result[1].is_nan() && result[3].is_nan() && result[5].is_nan()
        );
        equal(&[result[2], result[4]], &[0., 0.]);
        run(&mut group, 5., true);
        equal(&run(&mut group, 5., true)[..6], &[0.; 6]);
    }

    #[test]
    fn compilation_shares_history_across_kinds_windows_and_warmup_only_for_same_input() {
        let id = FeatureId::new;
        let mut definitions = Vec::new();
        for (i, constructor) in [
            T::rolling_slope,
            T::rolling_path_length,
            T::rolling_positive_movement,
            T::rolling_negative_movement,
            T::rolling_direction_change_count,
            T::rolling_max_drawdown,
        ]
        .into_iter()
        .enumerate()
        {
            definitions.push(constructor(
                id("x"),
                FeatureId::new(format!("out{i}")),
                i + 1,
                WarmupPolicy::FirstValue,
            ));
        }
        definitions.push(T::rolling_slope(
            id("x"),
            id("full"),
            2,
            WarmupPolicy::FullWindow,
        ));
        definitions.push(T::rolling_slope(
            id("y"),
            id("other"),
            2,
            WarmupPolicy::FirstValue,
        ));
        let groups = compile(&definitions, &[id("x"), id("y")]);
        assert_eq!(groups.len(), 2);
        let group = groups
            .iter()
            .find_map(|g| match g {
                Transformer::Trajectory(g) if g.input == 0 => Some(g),
                _ => None,
            })
            .unwrap();
        assert_eq!(group.history.capacity, 6);
        assert_eq!(group.windows.len(), 6);
        assert_eq!(group.windows[1].outputs.len(), 2);
    }
}

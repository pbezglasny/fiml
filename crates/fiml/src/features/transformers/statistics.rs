//! Shared circular summaries for exact extrema and stable population variance.

use super::Evaluation;
use crate::{
    FeatureVector, HeapRingBuffer, InvalidTransformationDefinitionError, RingBuffer, WarmupPolicy,
};

/// Selects the reported statistic from a shared window summary.
#[derive(Clone, Copy)]
pub(super) enum StatisticKind {
    Std,
    Min,
    Max,
    Range,
}

/// Means are relative to an observed origin to preserve small differences in large values.
#[derive(Clone, Copy, Default)]
struct Summary {
    count: usize,
    origin: f64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Summary {
    fn value(value: f64) -> Self {
        Self {
            count: 1,
            origin: value,
            min: value,
            max: value,
            ..Self::default()
        }
    }

    fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }
        let count = self.count + other.count;
        let weight = other.count as f64 / count as f64;
        let delta = (other.origin - self.origin) + (other.mean - self.mean);
        Self {
            count,
            origin: self.origin,
            mean: self.mean + delta * weight,
            m2: self.m2 + other.m2 + delta * delta * (self.count as f64 * weight),
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn statistic(self, kind: StatisticKind) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let value = match kind {
            StatisticKind::Std if self.m2.is_finite() => {
                (self.m2 / self.count as f64).max(0.).sqrt()
            }
            StatisticKind::Std => f64::NAN,
            StatisticKind::Min => self.min,
            StatisticKind::Max => self.max,
            StatisticKind::Range => self.max - self.min,
        };
        if value.is_finite() { value } else { f64::NAN }
    }
}

fn capacity(window: u64, aggregation: Option<i64>) -> Option<usize> {
    let slots = aggregation.map_or(Some(window), |a| (window / a as u64).checked_add(2))?;
    usize::try_from(slots).ok()
}

pub(super) fn validate_capacity(
    window: u64,
    aggregation: Option<i64>,
) -> Result<(), InvalidTransformationDefinitionError> {
    capacity(window, aggregation)
        .and_then(usize::checked_next_power_of_two)
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<Summary>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .map(|_| ())
        .ok_or(InvalidTransformationDefinitionError::StatisticsHistoryTooLarge)
}

/// A flat summary tree whose leaves are circular history slots; updates never subtract moments.
struct History {
    tree: Box<[Summary]>,
    base: usize,
    capacity: usize,
    next: usize,
    len: usize,
}
impl History {
    fn new(capacity: usize) -> Self {
        let base = capacity.next_power_of_two();
        Self {
            tree: vec![Summary::default(); 2 * base].into_boxed_slice(),
            base,
            capacity,
            next: 0,
            len: 0,
        }
    }
    fn write(&mut self, index: usize, value: Summary) {
        let mut node = self.base + index;
        self.tree[node] = value;
        while node > 1 {
            node /= 2;
            self.tree[node] = self.tree[node * 2].merge(self.tree[node * 2 + 1]);
        }
    }
    fn push(&mut self, value: f64, same_bucket: bool) {
        if same_bucket {
            let index = (self.next + self.capacity - 1) % self.capacity;
            self.write(
                index,
                self.tree[self.base + index].merge(Summary::value(value)),
            );
        } else {
            self.write(self.next, Summary::value(value));
            self.next = (self.next + 1) % self.capacity;
            self.len = (self.len + 1).min(self.capacity);
        }
    }
    fn query(&self, mut start: usize, mut end: usize) -> Summary {
        start += self.base;
        end += self.base;
        let (mut left, mut right) = (Summary::default(), Summary::default());
        while start < end {
            if start % 2 == 1 {
                left = left.merge(self.tree[start]);
                start += 1;
            }
            if end % 2 == 1 {
                end -= 1;
                right = self.tree[end].merge(right);
            }
            start /= 2;
            end /= 2;
        }
        left.merge(right)
    }
    fn newest(&self, count: usize) -> Summary {
        if count == 0 {
            return Summary::default();
        }
        let start = (self.next + self.capacity - count) % self.capacity;
        if start < self.next {
            self.query(start, self.next)
        } else {
            self.query(start, self.capacity)
                .merge(self.query(0, self.next))
        }
    }
}

/// Authored output position and warmup, with a cached value for unrelated events.
struct Output {
    kind: StatisticKind,
    warmup: WarmupPolicy,
    index: usize,
    visible: f64,
}
/// One range query serving every output with this window width.
struct Window {
    width: u64,
    outputs: Box<[Output]>,
}

/// One input's shared sample history or all-observation timed buckets.
pub(crate) struct StatisticsTransformer {
    input: usize,
    windows: Box<[Window]>,
    history: History,
    aggregation: Option<i64>,
    buckets: Option<HeapRingBuffer<i64>>,
    clock: Option<i64>,
    first: Option<i64>,
    finite: bool,
}

impl StatisticsTransformer {
    pub(super) fn new(
        input: usize,
        aggregation: Option<i64>,
        outputs: Vec<(StatisticKind, u64, WarmupPolicy, usize)>,
    ) -> Self {
        let size = capacity(outputs.iter().map(|o| o.1).max().unwrap(), aggregation).unwrap();
        let mut groups = std::collections::BTreeMap::<u64, Vec<Output>>::new();
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
            aggregation,
            buckets: aggregation.map(|_| HeapRingBuffer::new(size)),
            clock: None,
            first: None,
            finite: false,
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
        let source_observed = observed[self.input];
        let now = if self.aggregation.is_some() {
            evaluation.timestamps[self.input]
        } else {
            None
        };
        let refreshed = if self.aggregation.is_some() {
            evaluation.advance && now.is_some() && (source_observed || now > self.clock)
        } else {
            source_observed
        };
        if refreshed {
            self.clock = now;
            if source_observed {
                self.finite = input[self.input].is_finite();
                if self.finite {
                    let mut same_bucket = false;
                    if let Some(aggregation) = self.aggregation {
                        let now = now.unwrap();
                        self.first.get_or_insert(now);
                        let bucket = now.div_euclid(aggregation);
                        let buckets = self.buckets.as_mut().unwrap();
                        same_bucket = buckets.peek_back() == Some(&bucket);
                        if !same_bucket {
                            buckets.push_back(bucket);
                        }
                    }
                    self.history.push(input[self.input], same_bucket);
                }
            }
            for window in &mut self.windows {
                let (count, ready) = if let Some(aggregation) = self.aggregation {
                    let now = now.unwrap() as i128;
                    let buckets = self.buckets.as_ref().unwrap();
                    let (mut lo, mut hi) = (0, buckets.len());
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let start =
                            *buckets.peek_front_at(mid).unwrap() as i128 * aggregation as i128;
                        if start + window.width as i128 <= now {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    (
                        buckets.len() - lo,
                        self.first
                            .is_some_and(|first| now - first as i128 >= window.width as i128),
                    )
                } else {
                    (
                        self.history.len.min(window.width as usize),
                        self.history.len as u64 >= window.width,
                    )
                };
                let summary = self.history.newest(count);
                for out in &mut window.outputs {
                    out.visible =
                        if self.finite && (ready || out.warmup == WarmupPolicy::FirstValue) {
                            summary.statistic(out.kind)
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

    fn equal(actual: f64, expected: f64) {
        assert!(
            actual == expected
                || actual.is_nan() && expected.is_nan()
                || (actual - expected).abs() < 1e-10,
            "{actual} != {expected}"
        );
    }
    fn oracle(values: &[f64]) -> [f64; 4] {
        if values.is_empty() {
            return [f64::NAN; 4];
        }
        let origin = values[0];
        let mean = values.iter().map(|x| x - origin).sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|x| ((x - origin) - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let min = values.iter().copied().reduce(f64::min).unwrap();
        let max = values.iter().copied().reduce(f64::max).unwrap();
        [variance.sqrt(), min, max, max - min]
    }
    fn run(
        group: &mut StatisticsTransformer,
        value: f64,
        observed: bool,
        now: Option<i64>,
        advance: bool,
    ) -> Vec<f64> {
        let mut out = ArrayFeatureVector::<16>::new();
        let mut observed_out = [false; 16];
        group.apply(
            &[value],
            &[observed],
            &mut out,
            &mut observed_out,
            Evaluation {
                timestamps: &[now],
                global_timestamp: now,
                advance,
            },
        );
        out.values().to_vec()
    }
    fn outputs(width: u64, warmup: WarmupPolicy) -> Vec<(StatisticKind, u64, WarmupPolicy, usize)> {
        [
            StatisticKind::Std,
            StatisticKind::Min,
            StatisticKind::Max,
            StatisticKind::Range,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, kind)| (kind, width, warmup, i))
        .collect()
    }

    #[test]
    fn shares_history_across_kinds_windows_and_warmup_but_not_inputs_or_aggregation() {
        let id = FeatureId::new;
        let ms = std::time::Duration::from_millis;
        let groups = compile(
            &[
                T::rolling_std(id("x"), id("std"), 2, WarmupPolicy::FullWindow),
                T::rolling_min(id("x"), id("min"), 7, WarmupPolicy::FirstValue),
                T::rolling_range(id("x"), id("range"), 2, WarmupPolicy::FirstValue),
                T::rolling_max(id("y"), id("max"), 2, WarmupPolicy::FirstValue),
                T::rolling_std_timed(id("x"), id("t1"), ms(4), WarmupPolicy::FullWindow),
                T::rolling_range_timed(id("x"), id("t2"), ms(8), WarmupPolicy::FirstValue),
                T::rolling_min_timed_with_aggregation(
                    id("x"),
                    id("t3"),
                    ms(8),
                    ms(2),
                    WarmupPolicy::FirstValue,
                ),
            ],
            &[id("x"), id("y")],
        );
        assert_eq!(groups.len(), 4);
        let sample = groups
            .iter()
            .find_map(|g| match g {
                Transformer::Statistics(g) if g.input == 0 && g.aggregation.is_none() => Some(g),
                _ => None,
            })
            .unwrap();
        assert_eq!(sample.history.capacity, 7);
        assert_eq!(sample.windows.len(), 2);
    }

    #[test]
    fn sample_and_timed_wraparound_match_oracle_for_nearly_constant_large_values() {
        for aggregation in [None, Some(3)] {
            let mut group =
                StatisticsTransformer::new(0, aggregation, outputs(21, WarmupPolicy::FirstValue));
            let mut observations = Vec::new();
            for i in 0..5000_i64 {
                let time = i * 2 / 3 - 100;
                let value = 1e12 + ((i * 17) % 23) as f64 / 1024.;
                observations.push((time, value));
                let values: Vec<_> = if let Some(a) = aggregation {
                    observations
                        .iter()
                        .filter(|(t, _)| t.div_euclid(a) * a + 21 > time)
                        .map(|(_, v)| *v)
                        .collect()
                } else {
                    observations
                        .iter()
                        .rev()
                        .take(21)
                        .map(|(_, v)| *v)
                        .collect()
                };
                let actual = run(&mut group, value, true, Some(time), true);
                for (actual, expected) in actual.iter().zip(oracle(&values)) {
                    equal(*actual, expected);
                }
            }
        }
    }

    #[test]
    fn missing_inputs_warmup_and_clock_only_expiry() {
        for aggregation in [None, Some(1)] {
            let mut group =
                StatisticsTransformer::new(0, aggregation, outputs(2, WarmupPolicy::FullWindow));
            assert!(run(&mut group, 2., true, Some(0), true)[0].is_nan());
            assert!(run(&mut group, f64::INFINITY, true, Some(1), true)[0].is_nan());
            assert!(run(&mut group, 0., false, Some(1), true)[0].is_nan());
            let actual = run(&mut group, 6., true, Some(2), true);
            for (a, e) in actual.iter().zip(if aggregation.is_some() {
                [0., 6., 6., 0.]
            } else {
                [2., 2., 6., 4.]
            }) {
                equal(*a, e);
            }
            let retained = run(&mut group, 99., true, Some(3), false);
            if aggregation.is_some() {
                equal(retained[1], 6.);
            }
            let expired = run(&mut group, 0., false, Some(4), true);
            if aggregation.is_some() {
                assert!(expired[..4].iter().all(|v| v.is_nan()));
            }
        }
    }

    #[test]
    fn all_values_in_bucket_contribute_and_extreme_timestamps_do_not_overflow() {
        for start in [i64::MIN, -1, 0, i64::MAX - 100] {
            let mut group =
                StatisticsTransformer::new(0, Some(10), outputs(20, WarmupPolicy::FirstValue));
            run(&mut group, 2., true, Some(start), true);
            let row = run(&mut group, 10., true, Some(start), true);
            for (a, e) in row.iter().zip([4., 2., 10., 8.]) {
                equal(*a, e);
            }
            assert!(run(&mut group, 0., false, Some(start + 20), true)[0].is_nan());
            let row = run(&mut group, 3., true, Some(start + 20), true);
            for (a, e) in row.iter().zip([0., 3., 3., 0.]) {
                equal(*a, e);
            }
        }
    }

    #[test]
    fn overflowing_results_recover_after_eviction_and_one_sample_has_zero_dispersion() {
        let mut group = StatisticsTransformer::new(0, None, outputs(2, WarmupPolicy::FirstValue));
        equal(run(&mut group, f64::MAX, true, None, true)[0], 0.);
        let row = run(&mut group, -f64::MAX, true, None, true);
        assert!(row[0].is_nan() && row[3].is_nan());
        equal(row[1], -f64::MAX);
        run(&mut group, 5., true, None, true);
        let row = run(&mut group, 5., true, None, true);
        for (a, e) in row.iter().zip([0., 5., 5., 0.]) {
            equal(*a, e);
        }
    }
}

//! Experimental level-two signatures of piecewise-linear (observation index, value) paths.

use super::history::{History, Merge, valid_capacity};
use crate::{FeatureVector, WarmupPolicy};

/// Tensor levels keep channel order explicit without exposing a general-dimensional API.
#[derive(Clone, Copy, Default)]
struct Signature {
    one: [f64; 2],
    two: [[f64; 2]; 2],
}

impl Signature {
    fn segment(delta: [f64; 2]) -> Self {
        Self {
            one: delta,
            two: delta.map(|left| delta.map(|right| (0.5 * left) * right)),
        }
    }

    fn then(self, other: Self) -> Self {
        let mut result = Self::default();
        for i in 0..2 {
            result.one[i] = self.one[i] + other.one[i];
            for j in 0..2 {
                // Chen's identity: S2(AB) = S2(A) + S1(A) tensor S1(B) + S2(B).
                result.two[i][j] = self.two[i][j] + self.one[i] * other.one[j] + other.two[i][j];
            }
        }
        result
    }
}

/// Endpoints supply the connecting segment when adjacent observation ranges are composed.
#[derive(Clone, Copy, Default)]
struct Summary {
    count: usize,
    first: f64,
    last: f64,
    signature: Signature,
}

impl Summary {
    fn value(value: f64) -> Self {
        Self {
            count: 1,
            first: value,
            last: value,
            signature: Signature::default(),
        }
    }

    fn coordinates(self) -> [f64; 6] {
        let inverse_time = 1. / self.count.saturating_sub(1).max(1) as f64;
        let Signature { one, two } = self.signature;
        // Endpoint formulas avoid cancellation after large excursions return near the start.
        let change = self.last - self.first;
        [
            one[0] * inverse_time,
            change,
            two[0][0] * inverse_time * inverse_time,
            two[0][1] * inverse_time,
            two[1][0] * inverse_time,
            (0.5 * change) * change,
        ]
        .map(|value| if value.is_finite() { value } else { f64::NAN })
    }
}

impl Merge for Summary {
    fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }
        Self {
            count: self.count + other.count,
            first: self.first,
            last: other.last,
            signature: self
                .signature
                .then(Signature::segment([1., other.first - self.last]))
                .then(other.signature),
        }
    }
}

pub(crate) fn validate_window(window: usize) -> Result<(), String> {
    if window == 0 || !valid_capacity::<Summary>(window) {
        return Err(
            "path signature window must be positive and fit addressable history storage".into(),
        );
    }
    Ok(())
}

/// Retains a finite-observation window and composes its signature without update allocations.
pub(crate) struct PathSignature {
    pub(crate) input: usize,
    history: History<Summary>,
    warmup: WarmupPolicy,
    visible: [f64; 6],
}

impl PathSignature {
    pub(crate) fn new(input: usize, window: usize, warmup: WarmupPolicy) -> Self {
        Self {
            input,
            history: History::new(window),
            warmup,
            visible: [f64::NAN; 6],
        }
    }

    pub(crate) fn apply<V: FeatureVector>(
        &mut self,
        value: f64,
        observed: bool,
        output: &mut V,
        output_observed: &mut [bool],
    ) {
        if observed {
            if value.is_finite() {
                self.history.push(Summary::value(value), false);
            }
            self.visible = if value.is_finite()
                && (self.history.len == self.history.capacity
                    || self.warmup == WarmupPolicy::FirstValue)
            {
                self.history.newest(self.history.len).coordinates()
            } else {
                [f64::NAN; 6]
            };
        }
        for (index, value) in self.visible.iter().copied().enumerate() {
            output.set_value_at(index, value);
            output_observed[index] = observed;
        }
    }
}

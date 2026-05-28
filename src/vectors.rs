use std::{cell::Cell, fmt::Display};

use crate::Float;

pub struct Handler<'a, F: Float> {
    cell: &'a Cell<F>,
    idx: usize,
}

impl<'a, F: Float> Handler<'a, F> {
    pub fn set_value(&self, value: F) {
        self.cell.set(value);
    }

    pub fn get_index(&self) -> usize {
        self.idx
    }
}

impl<F: Float + Display> Display for Handler<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Handler(idx: {}, current val: {})",
            self.idx,
            self.cell.get(),
        )
    }
}

pub trait FeatureVector<F>
where
    F: Float,
{
    fn values(&self) -> &[F];

    fn next_handler<'a>(&'a mut self) -> Handler<'a, F>;
}

pub struct ArrayFeatureVector<F: Float, const N: usize> {
    data: [Cell<F>; N],
    next_idx: usize,
}

impl<F: Float, const N: usize> ArrayFeatureVector<F, N> {
    pub fn new() -> Self {
        Self {
            data: [const { Cell::new(F::ZERO) }; N],
            next_idx: 0,
        }
    }
}

impl<F: Float, const N: usize> Default for ArrayFeatureVector<F, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, const N: usize> FeatureVector<F> for ArrayFeatureVector<F, N> {
    fn values(&self) -> &[F] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast::<F>(), N) }
    }

    fn next_handler<'a>(&'a mut self) -> Handler<'a, F> {
        let idx = self.next_idx;
        if idx >= N {
            panic!("Exceeded maximum number of handlers for this feature vector");
        }
        self.next_idx += 1;
        Handler {
            cell: &self.data[idx],
            idx,
        }
    }
}

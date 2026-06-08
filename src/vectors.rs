use std::cell::Cell;

use crate::Float;

pub trait FeatureVector {
    type Float: Float;

    fn values(&self) -> &[Self::Float];

    fn capacity(&self) -> usize;

    fn set_value_at(&mut self, index: usize, value: Self::Float);

    fn set_values(&mut self, indexes: &[usize], values: &[Self::Float]) {
        for (index, value) in indexes.iter().zip(values.iter()) {
            self.set_value_at(*index, *value);
        }
    }
}

pub struct ArrayFeatureVector<F: Float, const N: usize> {
    data: [Cell<F>; N],
}

impl<F: Float, const N: usize> ArrayFeatureVector<F, N> {
    pub fn new() -> Self {
        Self {
            data: [const { Cell::new(F::ZERO) }; N],
        }
    }

    pub fn value_at(&self, index: usize) -> Option<F> {
        if index < N {
            Some(self.data[index].get())
        } else {
            None
        }
    }
}

impl<F: Float, const N: usize> Default for ArrayFeatureVector<F, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, const N: usize> FeatureVector for ArrayFeatureVector<F, N> {
    type Float = F;

    fn values(&self) -> &[F] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast::<F>(), N) }
    }

    fn capacity(&self) -> usize {
        N
    }

    fn set_value_at(&mut self, index: usize, value: F) {
        self.data[index].set(value);
    }
}

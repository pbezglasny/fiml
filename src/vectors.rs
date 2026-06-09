use std::cell::Cell;

use crate::Float;

pub trait FeatureVector {
    type F: Float;

    fn value_at(&self, index: usize) -> Option<Self::F>;

    fn values(&self) -> &[Self::F];

    fn capacity(&self) -> usize;

    fn set_value_at(&mut self, index: usize, value: Self::F);

    fn set_values(&mut self, indexes: &[usize], values: &[Self::F]) {
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
}

impl<F: Float, const N: usize> Default for ArrayFeatureVector<F, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, const N: usize> FeatureVector for ArrayFeatureVector<F, N> {
    type F = F;

    fn value_at(&self, index: usize) -> Option<F> {
        if index < N {
            Some(self.data[index].get())
        } else {
            None
        }
    }
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

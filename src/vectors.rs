use crate::{FimlError, Float, Result};

pub trait FeatureVector {
    type F: Float;

    fn value_at(&self, index: usize) -> Option<Self::F>;

    fn values(&self) -> &[Self::F];

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn set_value_at(&mut self, index: usize, value: Self::F);

    fn try_set_value_at(&mut self, index: usize, value: Self::F) -> Result<()> {
        if index >= self.len() {
            return Err(FimlError::InvalidArgument(format!(
                "index {} is out of bounds for feature vector of len {}",
                index,
                self.len()
            )));
        }
        self.set_value_at(index, value);
        Ok(())
    }

    fn set_values_range(&mut self, insert_index_start: usize, size: usize, values: &[Self::F]) {
        for (i, value) in values[..size].iter().enumerate() {
            self.set_value_at(insert_index_start + i, *value);
        }
    }

    fn try_set_values_range(
        &mut self,
        insert_index_start: usize,
        size: usize,
        values: &[Self::F],
    ) -> Result<()> {
        if size > values.len() {
            return Err(FimlError::InvalidArgument(format!(
                "Size {} is greater than the number of provided values {}",
                size,
                values.len()
            )));
        }
        let Some(end) = insert_index_start.checked_add(size) else {
            return Err(FimlError::InvalidArgument(format!(
                "range starting at {} with size {} overflows usize",
                insert_index_start, size
            )));
        };
        if end > self.len() {
            return Err(FimlError::InvalidArgument(format!(
                "range {}..{} is out of bounds for feature vector of len {}",
                insert_index_start,
                end,
                self.len()
            )));
        }
        self.set_values_range(insert_index_start, size, values);
        Ok(())
    }
}

pub struct ArrayFeatureVector<F: Float, const N: usize> {
    data: [F; N],
}

impl<F: Float, const N: usize> ArrayFeatureVector<F, N> {
    pub fn new() -> Self {
        Self { data: [F::ZERO; N] }
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
            Some(self.data[index])
        } else {
            None
        }
    }
    fn values(&self) -> &[F] {
        &self.data
    }

    fn len(&self) -> usize {
        N
    }

    fn set_value_at(&mut self, index: usize, value: F) {
        self.data[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_value_at_writes_without_result() {
        let mut values = ArrayFeatureVector::<f64, 2>::new();

        values.set_value_at(1, 4.0);

        assert_eq!(values.value_at(1), Some(4.0));
    }

    #[test]
    fn try_set_value_at_rejects_out_of_bounds_index() {
        let mut values = ArrayFeatureVector::<f64, 2>::new();

        let result = values.try_set_value_at(2, 4.0);

        assert!(result.is_err());
    }

    #[test]
    fn try_set_values_range_writes_valid_range() {
        let mut values = ArrayFeatureVector::<f64, 3>::new();

        values.try_set_values_range(1, 2, &[4.0, 5.0]).unwrap();

        assert_eq!(values.values(), &[0.0, 4.0, 5.0]);
    }

    #[test]
    fn try_set_values_range_rejects_out_of_bounds_range() {
        let mut values = ArrayFeatureVector::<f64, 3>::new();

        let result = values.try_set_values_range(2, 2, &[4.0, 5.0]);

        assert!(result.is_err());
    }
}

use crate::{FimlError, InvalidArgumentError, Result};

/// Abstraction layer for feature vector.
/// It supposed to set and get feature values
pub trait FeatureVector {
    /// Return value value at index
    /// Zero based indicies
    fn value_at(&self, index: usize) -> Option<f64>;

    /// Return all values of feature vector of capacity length
    fn values(&self) -> &[f64];

    /// Return total capacity of feature vector
    fn capacity(&self) -> usize;

    /// Return length of underlying collection of feaures.
    /// Len supposed to be less of equal capacity
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn set_value_at(&mut self, index: usize, value: f64);

    fn try_set_value_at(&mut self, index: usize, value: f64) -> Result<()> {
        if index >= self.len() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::FeatureVectorIndexOutOfBounds {
                    index,
                    length: self.len(),
                },
            ));
        }
        self.set_value_at(index, value);
        Ok(())
    }

    fn set_values_range(&mut self, insert_index_start: usize, size: usize, values: &[f64]) {
        for (i, value) in values[..size].iter().enumerate() {
            self.set_value_at(insert_index_start + i, *value);
        }
    }

    fn try_set_values_range(
        &mut self,
        insert_index_start: usize,
        size: usize,
        values: &[f64],
    ) -> Result<()> {
        if size > values.len() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::SourceValuesTooShort {
                    requested: size,
                    available: values.len(),
                },
            ));
        }
        let Some(end) = insert_index_start.checked_add(size) else {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::FeatureVectorRangeOverflow {
                    start: insert_index_start,
                    size,
                },
            ));
        };
        if end > self.capacity() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::FeatureVectorRangeOutOfBounds {
                    start: insert_index_start,
                    end,
                    capacity: self.capacity(),
                },
            ));
        }
        self.set_values_range(insert_index_start, size, values);
        Ok(())
    }
}

pub struct ArrayFeatureVector<const N: usize> {
    data: [f64; N],
    length: usize,
}

impl<const N: usize> ArrayFeatureVector<N> {
    /// Create new feature vector of capacity and length of N
    pub fn new() -> Self {
        Self::new_of_length(N)
    }

    /// Create new feature vector of capacity of N and provided length
    pub fn new_of_length(length: usize) -> Self {
        Self {
            data: [0.0; N],
            length,
        }
    }
}

impl<const N: usize> Default for ArrayFeatureVector<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> FeatureVector for ArrayFeatureVector<N> {
    fn value_at(&self, index: usize) -> Option<f64> {
        if index < N {
            Some(self.data[index])
        } else {
            None
        }
    }
    fn values(&self) -> &[f64] {
        &self.data
    }

    fn capacity(&self) -> usize {
        N
    }

    fn len(&self) -> usize {
        self.length
    }

    fn set_value_at(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }
}

/// Heap-backed [`FeatureVector`] whose cell count is chosen at runtime.
///
/// Use this when the number of features is not known at compile time (for
/// example when an engine is built from a deserialized spec). For a
/// compile-time fixed size prefer [`ArrayFeatureVector`], which avoids the heap
/// allocation.
pub struct VecFeatureVector {
    data: Vec<f64>,
    length: usize,
}

impl VecFeatureVector {
    /// Create feature vector of provided capacity and leghth of capacity
    pub fn new(capacity: usize) -> Self {
        Self::new_of_length(capacity, capacity)
    }

    /// Create feature vector of provided capacity and length
    pub fn new_of_length(capacity: usize, length: usize) -> Self {
        assert!(length <= capacity);
        Self {
            data: vec![0.0; capacity],
            length,
        }
    }
}

impl FeatureVector for VecFeatureVector {
    fn value_at(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }

    fn values(&self) -> &[f64] {
        &self.data
    }

    fn capacity(&self) -> usize {
        self.data.len()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn set_value_at(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_value_at_writes_without_result() {
        let mut values = ArrayFeatureVector::<2>::new();

        values.set_value_at(1, 4.0);

        assert_eq!(values.value_at(1), Some(4.0));
    }

    #[test]
    fn try_set_value_at_rejects_out_of_bounds_index() {
        let mut values = ArrayFeatureVector::<2>::new();

        let error = values.try_set_value_at(2, 4.0).unwrap_err();

        assert!(matches!(
            error,
            FimlError::InvalidArgument(InvalidArgumentError::FeatureVectorIndexOutOfBounds {
                index: 2,
                length: 2
            })
        ));
    }

    #[test]
    fn try_set_values_range_writes_valid_range() {
        let mut values = ArrayFeatureVector::<3>::new();

        values.try_set_values_range(1, 2, &[4.0, 5.0]).unwrap();

        assert_eq!(values.values(), &[0.0, 4.0, 5.0]);
    }

    #[test]
    fn try_set_values_range_rejects_out_of_bounds_range() {
        let mut values = ArrayFeatureVector::<3>::new();

        let error = values.try_set_values_range(2, 2, &[4.0, 5.0]).unwrap_err();

        assert!(matches!(
            error,
            FimlError::InvalidArgument(InvalidArgumentError::FeatureVectorRangeOutOfBounds {
                start: 2,
                end: 4,
                capacity: 3
            })
        ));
    }

    #[test]
    fn vec_feature_vector_starts_zeroed_with_runtime_len() {
        let values = VecFeatureVector::new(3);

        assert_eq!(values.capacity(), 3);
        assert_eq!(values.values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn vec_feature_vector_writes_and_reads_by_index() {
        let mut values = VecFeatureVector::new(2);

        values.set_value_at(1, 4.0);

        assert_eq!(values.value_at(1), Some(4.0));
        assert_eq!(values.value_at(2), None);
    }

    #[test]
    fn vec_feature_vector_rejects_out_of_bounds_via_try() {
        let mut values = VecFeatureVector::new(2);

        let result = values.try_set_value_at(2, 4.0);

        assert!(result.is_err());
    }

    #[test]
    fn vec_feature_vector_capacity_is_model_width_not_allocator_capacity() {
        let mut data = Vec::with_capacity(16);
        data.resize(2, 0.0);
        let values = VecFeatureVector { data, length: 2 };

        assert_eq!(values.capacity(), 2);
    }
}

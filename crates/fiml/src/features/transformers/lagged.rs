use crate::{FeatureVector, HeapRingBuffer, RingBuffer};

/// Emits multiple event lags of one raw feature using one shared history buffer.
/// A lag of `N` first writes on call `N + 1`; until then its output is untouched.
pub struct LaggedFeature {
    input_index: usize,
    // Each pair contains a positive lag window and its output index.
    outputs: Box<[(usize, usize)]>,
    buffer: HeapRingBuffer<f64>,
}

impl LaggedFeature {
    /// Allocates history for the largest window and preserves window/output pairs.
    /// Panics for empty windows, zero lags, or mismatched window/output counts.
    pub fn new(input_index: usize, windows: Vec<usize>, output_indices: Box<[usize]>) -> Self {
        assert!(!windows.is_empty(), "At least one lag window is required");
        assert_eq!(
            windows.len(),
            output_indices.len(),
            "Lag window and output counts must match"
        );
        assert!(
            windows.iter().all(|&window| window > 0),
            "Lag windows must be positive"
        );
        let max_window_size = *windows.iter().max().unwrap();
        let outputs = windows.into_iter().zip(output_indices).collect();
        Self {
            input_index,
            outputs,
            buffer: HeapRingBuffer::new(max_window_size),
        }
    }

    pub fn apply<V: FeatureVector>(&mut self, input_values: &[f64], output_vector: &mut V) {
        for (window, output_idx) in &self.outputs {
            if let Some(value) = self.buffer.peek_back_at(*window - 1) {
                output_vector.set_value_at(*output_idx, *value);
            }
        }
        let _ = self.buffer.push_back(input_values[self.input_index]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrayFeatureVector;

    #[test]
    fn shares_history_across_unsorted_windows_and_duplicate_lags() {
        let mut transformer = LaggedFeature::new(1, vec![3, 1, 2, 1], Box::new([4, 1, 3, 2]));
        let mut output = ArrayFeatureVector::<6>::new();
        output.set_values_range(0, 6, &[99.0; 6]);
        let inputs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        for (index, input) in inputs.into_iter().enumerate() {
            transformer.apply(&[-1.0, input], &mut output);
            for (window, output_index) in [(3, 4), (1, 1), (2, 3), (1, 2)] {
                let expected = index.checked_sub(window).map_or(99.0, |i| inputs[i]);
                assert_eq!(output.values()[output_index], expected);
            }
            assert_eq!(output.values()[0], 99.0);
            assert_eq!(output.values()[5], 99.0);
        }
        assert_eq!(transformer.buffer.capacity(), 3);
    }

    #[test]
    fn delays_selected_input_by_exactly_the_lag_window() {
        let inputs = [10.0, 10.0, -5.0, 0.0, 30.0, 40.0, 50.0, 60.0];
        for lag_window in [1, 3, 10] {
            let mut transformer = LaggedFeature::new(2, vec![lag_window], Box::new([1]));
            let mut output = ArrayFeatureVector::<3>::new();
            output.set_values_range(0, 3, &[111.0, -999.0, 333.0]);

            for (index, input) in inputs.into_iter().enumerate() {
                let raw_values = [7.0, 8.0, input];
                transformer.apply(&raw_values, &mut output);

                let expected = if index < lag_window {
                    -999.0
                } else {
                    inputs[index - lag_window]
                };
                assert_eq!(output.values(), &[111.0, expected, 333.0]);
                assert_eq!(raw_values, [7.0, 8.0, input]);
            }
        }
    }

    #[test]
    fn preserves_warmup_nan_and_delays_nan_inputs_without_skipping_them() {
        let mut transformer = LaggedFeature::new(0, vec![2, 1], Box::new([0, 1]));
        let mut output = ArrayFeatureVector::<2>::new();
        output.set_values_range(0, 2, &[f64::NAN; 2]);

        let cases = [
            (10.0, [f64::NAN, f64::NAN]),
            (f64::NAN, [f64::NAN, 10.0]),
            (30.0, [10.0, f64::NAN]),
            (40.0, [f64::NAN, 30.0]),
            (50.0, [30.0, 40.0]),
        ];
        for (input, expected) in cases {
            transformer.apply(&[input], &mut output);
            for (actual, expected) in output.values().iter().zip(expected) {
                assert!((actual.is_nan() && expected.is_nan()) || *actual == expected);
            }
        }
    }

    #[test]
    fn transformer_instances_keep_independent_histories() {
        let mut first = LaggedFeature::new(0, vec![1], Box::new([0]));
        let mut second = LaggedFeature::new(0, vec![1], Box::new([1]));
        let mut output = ArrayFeatureVector::<2>::new();

        first.apply(&[10.0], &mut output);
        second.apply(&[100.0], &mut output);
        first.apply(&[20.0], &mut output);
        second.apply(&[200.0], &mut output);

        assert_eq!(output.values(), &[10.0, 100.0]);
    }

    #[test]
    #[should_panic(expected = "Lag windows must be positive")]
    fn rejects_zero_lag() {
        LaggedFeature::new(0, vec![0, 2], Box::new([0, 1]));
    }

    #[test]
    #[should_panic(expected = "At least one lag window is required")]
    fn rejects_empty_windows() {
        LaggedFeature::new(0, vec![], Box::new([]));
    }

    #[test]
    #[should_panic(expected = "Lag window and output counts must match")]
    fn rejects_mismatched_output_count() {
        LaggedFeature::new(0, vec![1, 2], Box::new([0]));
    }
}

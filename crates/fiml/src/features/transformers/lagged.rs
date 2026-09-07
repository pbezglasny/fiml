use crate::FeatureVector;
use crate::HeapRingBuffer;
use crate::RingBuffer;

/// Lagged feature transformer: value in feature cell will be avaible after lag window events
/// and will be lagged for this number of events
pub struct LaggedFeature {
    input_index: usize,
    output_index: usize,
    buffer: HeapRingBuffer<f64>,
}

impl LaggedFeature {
    pub fn new(lag_window: usize, input_index: usize, output_index: usize) -> Self {
        Self {
            input_index,
            output_index,
            buffer: HeapRingBuffer::new(lag_window),
        }
    }

    pub fn apply<V: FeatureVector>(&mut self, raw_values: &[f64], model_vector: &mut V) {
        let old_value = self.buffer.push_back(raw_values[self.input_index]);
        if let Some(value) = old_value {
            model_vector.set_value_at(self.output_index, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrayFeatureVector;

    #[test]
    fn delays_selected_input_by_exactly_the_lag_window() {
        let inputs = [10.0, 10.0, -5.0, 0.0, 30.0, 40.0, 50.0, 60.0];
        for lag_window in [1, 3, 10] {
            let mut transformer = LaggedFeature::new(lag_window, 2, 1);
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
        let mut transformer = LaggedFeature::new(2, 0, 0);
        let mut output = ArrayFeatureVector::<1>::new();
        output.set_value_at(0, f64::NAN);

        let cases = [
            (10.0, f64::NAN),
            (f64::NAN, f64::NAN),
            (30.0, 10.0),
            (40.0, f64::NAN),
            (50.0, 30.0),
        ];
        for (input, expected) in cases {
            transformer.apply(&[input], &mut output);
            if expected.is_nan() {
                assert!(output.values()[0].is_nan());
            } else {
                assert_eq!(output.values()[0], expected);
            }
        }
    }

    #[test]
    fn transformer_instances_keep_independent_histories() {
        let mut first = LaggedFeature::new(1, 0, 0);
        let mut second = LaggedFeature::new(1, 0, 1);
        let mut output = ArrayFeatureVector::<2>::new();

        first.apply(&[10.0], &mut output);
        second.apply(&[100.0], &mut output);
        first.apply(&[20.0], &mut output);
        second.apply(&[200.0], &mut output);

        assert_eq!(output.values(), &[10.0, 100.0]);
    }

    #[test]
    #[should_panic(expected = "Ring buffer size must be greater than 0")]
    fn rejects_zero_lag() {
        LaggedFeature::new(0, 0, 0);
    }
}

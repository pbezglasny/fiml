use crate::FeatureVector;

/// Scales a raw scalar into model input using a precomputed inverse scale.
/// Resolved indexes and parameters keep execution allocation-free.
pub(crate) struct StandardScaleTransformer {
    input_index: usize,
    output_index: usize,
    mean: f64,
    inverse_scale: f64,
}

impl StandardScaleTransformer {
    /// Precomputes the inverse scale from parameters validated by the definition.
    pub(crate) fn new(input_index: usize, output_index: usize, mean: f64, scale: f64) -> Self {
        Self {
            input_index,
            output_index,
            mean,
            inverse_scale: 1.0 / scale,
        }
    }

    pub(crate) fn apply<V: FeatureVector>(&self, raw_values: &[f64], model_vector: &mut V) {
        model_vector.set_value_at(
            self.output_index,
            (raw_values[self.input_index] - self.mean) * self.inverse_scale,
        );
    }
}

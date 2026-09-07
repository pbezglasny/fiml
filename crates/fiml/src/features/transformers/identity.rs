use crate::FeatureVector;

/// Copies a raw scalar into model input using resolved indexes without allocation.
pub(crate) struct IdentityTransformer {
    input_index: usize,
    output_index: usize,
}

impl IdentityTransformer {
    pub(crate) fn new(input_index: usize, output_index: usize) -> Self {
        Self {
            input_index,
            output_index,
        }
    }

    pub(crate) fn apply<V: FeatureVector>(&self, raw_values: &[f64], model_vector: &mut V) {
        model_vector.set_value_at(self.output_index, raw_values[self.input_index]);
    }
}

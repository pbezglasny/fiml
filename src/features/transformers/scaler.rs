use crate::{FeatureVector, Float, features::transformers::Transformation};

pub struct StandardScaler<F: Float, V: FeatureVector<F = F>, const SIZE: usize> {
    input_idxes: [usize; SIZE],
    output_idxes: [usize; SIZE],
    mean: F,
    deviation: F,
    output: V,
}

impl<F: Float, V: FeatureVector<F = F>, const SIZE: usize> StandardScaler<F, V, SIZE> {
    pub fn new(
        input_idxes: [usize; SIZE],
        output_idxes: [usize; SIZE],
        mean: F,
        deviation: F,
        output: V,
    ) -> Self {
        Self {
            input_idxes,
            output_idxes,
            mean,
            deviation,
            output,
        }
    }
}

impl<F: Float, V: FeatureVector<F = F>, const SIZE: usize> Transformation
    for StandardScaler<F, V, SIZE>
{
    type F = F;
    type OutputVector = V;

    fn transform<I>(&mut self, input: &I)
    where
        I: FeatureVector<F = F>,
    {
        for (i, o) in self.input_idxes.iter().zip(self.output_idxes.iter()) {
            if let Some(value) = input.value_at(*i) {
                self.output
                    .set_value_at(*o, value.sub(self.mean).div(self.deviation));
            }
        }
    }

    fn output_values(&self) -> &Self::OutputVector {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn standard_scaler_scales_single_value() {
        let mut input = ArrayFeatureVector::<f64, 1>::new();
        input.set_value_at(0, 10.0);

        let mut scaler =
            StandardScaler::new([0], [0], 2.0, 4.0, ArrayFeatureVector::<f64, 1>::new());
        scaler.transform(&input);

        assert!(approx_eq(scaler.output_values().values()[0], 2.0));
    }

    #[test]
    fn standard_scaler_scales_multiple_values_with_index_remapping() {
        let mut input = ArrayFeatureVector::<f64, 3>::new();
        input.set_value_at(0, 6.0);
        input.set_value_at(2, 10.0);

        let mut scaler = StandardScaler::new(
            [0, 2],
            [1, 0],
            2.0,
            2.0,
            ArrayFeatureVector::<f64, 2>::new(),
        );
        scaler.transform(&input);

        assert!(approx_eq(scaler.output_values().values()[0], 4.0));
        assert!(approx_eq(scaler.output_values().values()[1], 2.0));
    }

    #[test]
    fn standard_scaler_skips_missing_input_index() {
        let input = ArrayFeatureVector::<f64, 1>::new();
        let mut output = ArrayFeatureVector::<f64, 1>::new();
        output.set_value_at(0, 42.0);

        let mut scaler = StandardScaler::new([1], [0], 2.0, 4.0, output);
        scaler.transform(&input);

        assert!(approx_eq(scaler.output_values().values()[0], 42.0));
    }
}

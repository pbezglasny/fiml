use crate::{
    Float,
    features::transformers::{TransformInput, TransformOutput, Transformation},
};

pub struct StandardScaler<F: Float, O: TransformOutput<F>, const SIZE: usize> {
    input_idxes: [usize; SIZE],
    output_idxes: [usize; SIZE],
    mean: F,
    deviation: F,
    output: O,
}

impl<F: Float, O: TransformOutput<F>, const SIZE: usize> StandardScaler<F, O, SIZE> {
    pub fn new(
        input_idxes: [usize; SIZE],
        output_idxes: [usize; SIZE],
        mean: F,
        deviation: F,
        output: O,
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

impl<F: Float, O: TransformOutput<F>, const SIZE: usize> TransformInput<F>
    for StandardScaler<F, O, SIZE>
{
    fn value_at(&self, index: usize) -> Option<F> {
        self.output.values().get(index).copied()
    }
}

impl<F: Float, O: TransformOutput<F>, const SIZE: usize> Transformation<F>
    for StandardScaler<F, O, SIZE>
{
    fn transform<I>(&mut self, input: &I)
    where
        I: TransformInput<F> + ?Sized,
    {
        for (i, o) in self.input_idxes.iter().zip(self.output_idxes.iter()) {
            if let Some(value) = input.value_at(*i) {
                self.output
                    .set_value_at(*o, value.sub(self.mean).div(self.deviation));
            }
        }
    }

    fn output_values(&self) -> &[F] {
        self.output.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::transformers::BuiltinTransfomers;
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

        assert!(approx_eq(Transformation::output_values(&scaler)[0], 2.0));
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

        assert!(approx_eq(Transformation::output_values(&scaler)[0], 4.0));
        assert!(approx_eq(Transformation::output_values(&scaler)[1], 2.0));
    }

    #[test]
    fn standard_scaler_skips_missing_input_index() {
        let input = ArrayFeatureVector::<f64, 1>::new();
        let mut output = ArrayFeatureVector::<f64, 1>::new();
        output.set_value_at(0, 42.0);

        let mut scaler = StandardScaler::new([1], [0], 2.0, 4.0, output);
        scaler.transform(&input);

        assert!(approx_eq(Transformation::output_values(&scaler)[0], 42.0));
    }

    #[test]
    fn builtin_transformer_dispatches_standard_scaler() {
        let mut input = ArrayFeatureVector::<f64, 2>::new();
        input.set_value_at(0, 6.0);
        input.set_value_at(1, 8.0);

        let mut transformer = BuiltinTransfomers::StandardScaler2(StandardScaler::new(
            [0, 1],
            [0, 1],
            2.0,
            2.0,
            ArrayFeatureVector::<f64, 2>::new(),
        ));
        transformer.transform(&input);

        assert!(approx_eq(
            Transformation::output_values(&transformer)[0],
            2.0
        ));
        assert!(approx_eq(
            Transformation::output_values(&transformer)[1],
            3.0
        ));
    }
}

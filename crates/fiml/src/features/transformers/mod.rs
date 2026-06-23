mod scaler;
use crate::{FeatureVector, Float};

pub use scaler::StandardScaler;

pub trait Transformation {
    type F: Float;
    type OutputVector: FeatureVector<F = Self::F>;

    fn transform<V>(&mut self, input: &V)
    where
        V: FeatureVector<F = Self::F>;

    fn output_values(&self) -> &Self::OutputVector;

    fn output_values_mut(&mut self) -> &mut Self::OutputVector;
}

pub struct ParallelTransformer<F, V, T, const NUM_TRANSFORMERS: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
    T: Transformation<F = F, OutputVector = V>,
{
    steps: [T; NUM_TRANSFORMERS],
    output_write_idx: [usize; NUM_TRANSFORMERS],
    num_transformers: usize,
    output: V,
}

impl<F, V, T, const NUM_TRANSFORMERS: usize> Transformation
    for ParallelTransformer<F, V, T, NUM_TRANSFORMERS>
where
    F: Float,
    V: FeatureVector<F = F>,
    T: Transformation<F = F, OutputVector = V>,
{
    type F = F;

    type OutputVector = V;

    fn transform<I>(&mut self, input: &I)
    where
        I: FeatureVector<F = Self::F>,
    {
        for i in 0..self.num_transformers {
            let transformer = &mut self.steps[i];
            transformer.transform(input);
            let output_values = transformer.output_values();
            let output_idx = self.output_write_idx[i];
            self.output
                .set_values_range(output_idx, output_values.len(), output_values.values());
        }
    }

    fn output_values(&self) -> &Self::OutputVector {
        &self.output
    }

    fn output_values_mut(&mut self) -> &mut Self::OutputVector {
        &mut self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct TestVector<const N: usize> {
        data: [Cell<f64>; N],
        len: usize,
    }

    impl<const N: usize> TestVector<N> {
        fn new(len: usize) -> Self {
            assert!(len <= N);
            Self {
                data: [const { Cell::new(0.0) }; N],
                len,
            }
        }
    }

    impl<const N: usize> FeatureVector for TestVector<N> {
        type F = f64;

        fn value_at(&self, index: usize) -> Option<Self::F> {
            if index < self.len {
                Some(self.data[index].get())
            } else {
                None
            }
        }

        fn values(&self) -> &[Self::F] {
            unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast::<f64>(), self.len) }
        }

        fn len(&self) -> usize {
            self.len
        }

        fn set_value_at(&mut self, index: usize, value: Self::F) {
            assert!(index < self.len);
            self.data[index].set(value);
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn parallel_transformer_runs_all_transformers_against_same_input_and_combines_outputs() {
        let mut input = TestVector::<4>::new(2);
        input.set_value_at(0, 4.0);
        input.set_value_at(1, 6.0);

        let mut transformer = ParallelTransformer {
            steps: [
                StandardScaler::new([0, 1], [0, 1], 1.0, 1.0, TestVector::<4>::new(2)),
                StandardScaler::new([0, 1], [0, 1], 2.0, 2.0, TestVector::<4>::new(2)),
            ],
            output_write_idx: [0, 2],
            num_transformers: 2,
            output: TestVector::<4>::new(4),
        };

        transformer.transform(&input);

        let output = transformer.output_values().values();
        assert!(approx_eq(output[0], 3.0));
        assert!(approx_eq(output[1], 5.0));
        assert!(approx_eq(output[2], 1.0));
        assert!(approx_eq(output[3], 2.0));
    }
}

// pub enum BuiltinTransfomers<F: Float> {
//     StandardScaler1(StandardScaler<F, ArrayFeatureVector<F, 1>, 1>),
//     StandardScaler2(StandardScaler<F, ArrayFeatureVector<F, 2>, 2>),
//     StandardScaler3(StandardScaler<F, ArrayFeatureVector<F, 3>, 3>),
// }

// impl<F: Float, V: FeatureVector<Float = F>> Transformation<F, V> for BuiltinTransfomers<F> {
//     fn transform(&mut self, input: &V) {
//         match self {
//             BuiltinTransfomers::StandardScaler1(s) => s.transform(input),
//             BuiltinTransfomers::StandardScaler2(s) => s.transform(input),
//             BuiltinTransfomers::StandardScaler3(s) => s.transform(input),
//         }
//     }
//
//     fn output_values(&self) -> &V {
//         match self {
//             BuiltinTransfomers::StandardScaler1(s) => s.output_values(),
//             BuiltinTransfomers::StandardScaler2(s) => s.output_values(),
//             BuiltinTransfomers::StandardScaler3(s) => s.output_values(),
//         }
//     }
// }

mod scaler;
use crate::{ArrayFeatureVector, Result};
use crate::{FeatureVector, Float};

pub use scaler::StandardScaler;

pub trait TransformInput<F: Float> {
    fn value_at(&self, index: usize) -> Option<F>;

    fn values<const N: usize>(&self, indexes: [usize; N]) -> Result<[F; N]> {
        let mut result = [F::ZERO; N];
        for i in 0..N {
            if let Some(value) = self.value_at(indexes[i]) {
                result[i] = value;
            } else {
                return Err(crate::FimlError::InvalidArgument(format!(
                    "index {} is out of bounds",
                    indexes[i]
                )));
            }
        }
        Ok(result)
    }
}

impl<F: Float> TransformInput<F> for [F] {
    fn value_at(&self, index: usize) -> Option<F> {
        self.get(index).copied()
    }
}

pub trait TransformOutput<F: Float>: FeatureVector<Float = F> {}

pub trait Transformation<F: Float>: TransformInput<F> {
    fn transform<I>(&mut self, input: &I)
    where
        I: TransformInput<F> + ?Sized;

    fn output_values(&self) -> &[F];
}

impl<F: Float, const N: usize> TransformInput<F> for ArrayFeatureVector<F, N> {
    fn value_at(&self, index: usize) -> Option<F> {
        self.value_at(index)
    }
}

impl<T: FeatureVector<Float = F>, F: Float> TransformOutput<F> for T {}

pub enum BuiltinTransfomers<F: Float> {
    StandardScaler1(StandardScaler<F, ArrayFeatureVector<F, 1>, 1>),
    StandardScaler2(StandardScaler<F, ArrayFeatureVector<F, 2>, 2>),
    StandardScaler3(StandardScaler<F, ArrayFeatureVector<F, 3>, 3>),
}

impl<F: Float> TransformInput<F> for BuiltinTransfomers<F> {
    fn value_at(&self, index: usize) -> Option<F> {
        match self {
            BuiltinTransfomers::StandardScaler1(s) => s.value_at(index),
            BuiltinTransfomers::StandardScaler2(s) => s.value_at(index),
            BuiltinTransfomers::StandardScaler3(s) => s.value_at(index),
        }
    }
}

impl<F: Float> Transformation<F> for BuiltinTransfomers<F> {
    fn transform<I>(&mut self, input: &I)
    where
        I: TransformInput<F> + ?Sized,
    {
        match self {
            BuiltinTransfomers::StandardScaler1(s) => s.transform(input),
            BuiltinTransfomers::StandardScaler2(s) => s.transform(input),
            BuiltinTransfomers::StandardScaler3(s) => s.transform(input),
        }
    }

    fn output_values(&self) -> &[F] {
        match self {
            BuiltinTransfomers::StandardScaler1(s) => s.output_values(),
            BuiltinTransfomers::StandardScaler2(s) => s.output_values(),
            BuiltinTransfomers::StandardScaler3(s) => s.output_values(),
        }
    }
}

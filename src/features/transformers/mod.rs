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

pub trait TransformOutput<F: Float>: FeatureVector<Float = F> {}

pub trait Transformation<F: Float> {
    fn update<I, O>(&mut self, input: &I, output: &mut O)
    where
        I: TransformInput<F>,
        O: TransformOutput<F>;
}

impl<F: Float, const N: usize> TransformInput<F> for ArrayFeatureVector<F, N> {
    fn value_at(&self, index: usize) -> Option<F> {
        self.value_at(index)
    }
}

impl<T: FeatureVector<Float = F>, F: Float> TransformOutput<F> for T {}

pub enum BuiltinTransfomers<F: Float> {
    StandardScaler1(StandardScaler<F, 1>),
    StandardScaler2(StandardScaler<F, 2>),
    StandardScaler3(StandardScaler<F, 3>),
}

impl<F: Float> Transformation<F> for BuiltinTransfomers<F> {
    fn update<I, O>(&mut self, input: &I, output: &mut O)
    where
        I: TransformInput<F>,
        O: TransformOutput<F>,
    {
        match self {
            BuiltinTransfomers::StandardScaler1(s) => s.update(input, output),
            BuiltinTransfomers::StandardScaler2(s) => s.update(input, output),
            BuiltinTransfomers::StandardScaler3(s) => s.update(input, output),
        }
    }
}

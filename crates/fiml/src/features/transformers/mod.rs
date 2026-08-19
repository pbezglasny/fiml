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

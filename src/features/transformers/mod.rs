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

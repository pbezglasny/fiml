use std::mem::MaybeUninit;

use crate::FeatureVector;
use crate::features::IndicatorFeatures;
use crate::features::transformers::Transformation;

pub struct Pipeline<I, T, O, const TRANSFORMER_SIZE: usize = 10>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
    O: FeatureVector<I::Float>,
{
    indicators: I,
    transformers: [MaybeUninit<T>; TRANSFORMER_SIZE],
    num_transformers: usize,
    transform_output: O,
}

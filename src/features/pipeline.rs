use std::mem::MaybeUninit;

use crate::features::IndicatorFeatureVector;

use crate::features::transformers::Transformation;
use crate::{Feature, FeatureVector, Float};

pub struct Pipeline<F, I, V, T, const INDICATOR_SIZE: usize, const TRANSFORMER_SIZE: usize>
where
    F: Float,
    I: Feature<F>,
    V: FeatureVector<F>,
    T: Transformation<F>,
{
    indicators: IndicatorFeatureVector<F, V, I, INDICATOR_SIZE>,
    transformers: [MaybeUninit<T>; TRANSFORMER_SIZE],
}

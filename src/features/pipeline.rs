use std::mem::MaybeUninit;

use crate::FeatureVector;
use crate::features::IndicatorFeatures;
use crate::features::transformers::Transformation;

pub struct Pipeline<I, T, O, const TRANSFORMER_SIZE: usize>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
    O: FeatureVector<Float = I::Float>,
{
    indicators: I,
    transformers: [MaybeUninit<T>; TRANSFORMER_SIZE],
    num_transformers: usize,
    transform_output: O,
}

macro_rules! impl_pipeline {
    ($($num_transformers:expr),*) => {
        $(
            impl<I, T, O> Pipeline<I, T, O, $num_transformers>
            where
                I: IndicatorFeatures,
                T: Transformation<I::Float>,
                O: FeatureVector<Float = I::Float>,
            {
                pub const fn new(indicators: I, transform_output: O) -> Self {
                    Self {
                        indicators,
                        transformers: unsafe { MaybeUninit::uninit().assume_init() },
                        num_transformers: 0,
                        transform_output,
                    }
                }
            }
        )*
    };
}

impl_pipeline!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

impl<I, T, O, const TRANSFORMER_SIZE: usize> Pipeline<I, T, O, TRANSFORMER_SIZE>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
    O: FeatureVector<Float = I::Float>,
{
}

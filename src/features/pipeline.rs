use std::mem::MaybeUninit;

use crate::features::IndicatorFeatures;
use crate::features::transformers::Transformation;
use crate::{Event, FeatureVector, FimlError};

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
    pub fn add_transformer(&mut self, transformer: T) -> Result<(), FimlError> {
        if self.num_transformers < TRANSFORMER_SIZE {
            self.transformers[self.num_transformers].write(transformer);
            self.num_transformers += 1;
            Ok(())
        } else {
            Err(FimlError::InvalidArgument(format!(
                "cannot add more than {} transformers",
                self.num_transformers
            )))
        }
    }
}

impl<I, T, O, const TRANSFORMER_SIZE: usize> Pipeline<I, T, O, TRANSFORMER_SIZE>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
    O: FeatureVector<Float = I::Float>,
{
    pub fn dispatch(&mut self, event: &Event<I::Float>) {
        self.indicators.dispatch(event);
        for i in 0..self.num_transformers {
            let transformer = unsafe { self.transformers[i].assume_init_mut() };
            transformer.transform(self.indicators.values(), &mut self.transform_output);
        }
    }
}

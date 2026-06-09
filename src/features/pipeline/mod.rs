use std::mem::MaybeUninit;

use crate::features::IndicatorFeatures;
use crate::features::transformers::Transformation;
use crate::{Event, FimlError};

pub struct Pipeline<I, T, const TRANSFORMER_SIZE: usize>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
{
    indicators: I,
    transformers: [MaybeUninit<T>; TRANSFORMER_SIZE],
    num_transformers: usize,
}

impl<I, T, const TRANSFORMER_SIZE: usize> Pipeline<I, T, TRANSFORMER_SIZE>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
{
    pub fn new(indicators: I) -> Self {
        Self {
            indicators,
            transformers: [const { MaybeUninit::uninit() }; TRANSFORMER_SIZE],
            num_transformers: 0,
        }
    }

    pub fn add_transformer(&mut self, transformer: T) -> Result<(), FimlError> {
        if self.num_transformers < TRANSFORMER_SIZE {
            self.transformers[self.num_transformers].write(transformer);
            self.num_transformers += 1;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                transformer_index = self.num_transformers - 1,
                transformer_count = self.num_transformers,
                transformer_capacity = TRANSFORMER_SIZE,
                transformer_type = std::any::type_name::<T>(),
                "added pipeline transformer"
            );
            Ok(())
        } else {
            Err(FimlError::InvalidArgument(format!(
                "cannot add more than {} transformers",
                self.num_transformers
            )))
        }
    }
}

impl<I, T, const TRANSFORMER_SIZE: usize> Drop for Pipeline<I, T, TRANSFORMER_SIZE>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
{
    fn drop(&mut self) {
        // SAFETY: the first `num_transformers` entries are initialized by
        // `add_transformer`.
        for slot in &mut self.transformers[..self.num_transformers] {
            unsafe { slot.assume_init_drop() };
        }
    }
}

impl<I, T, const TRANSFORMER_SIZE: usize> Pipeline<I, T, TRANSFORMER_SIZE>
where
    I: IndicatorFeatures,
    T: Transformation<I::Float>,
{
    pub fn dispatch(&mut self, event: &Event<I::Float>) {
        self.indicators.dispatch(event);
        if self.num_transformers == 0 {
            return;
        }

        let first = unsafe { self.transformers[0].assume_init_mut() };
        first.transform(self.indicators.values());

        for i in 1..self.num_transformers {
            let (previous, current) = self.transformers.split_at_mut(i);
            let input = unsafe { previous[i - 1].assume_init_ref() };
            let output = unsafe { current[0].assume_init_mut() };
            output.transform(input);
        }
    }

    pub fn values(&self) -> &[I::Float] {
        if self.num_transformers == 0 {
            self.indicators.values()
        } else {
            unsafe { self.transformers[self.num_transformers - 1].assume_init_ref() }
                .output_values()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::transformers::StandardScaler;
    use crate::{ArrayFeatureVector, FeatureVector, Ticker};

    struct TestIndicators {
        cells: ArrayFeatureVector<f64, 1>,
    }

    impl TestIndicators {
        fn new(value: f64) -> Self {
            let mut cells = ArrayFeatureVector::new();
            cells.set_value_at(0, value);
            Self { cells }
        }
    }

    impl IndicatorFeatures for TestIndicators {
        type Float = f64;

        fn dispatch(&mut self, _event: &Event<Self::Float>) {}

        fn values(&self) -> &[Self::Float] {
            self.cells.values()
        }

        fn index_of(&self, _ticker: Ticker, _name: &str) -> Option<usize> {
            None
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn values_returns_indicator_output_without_transformers() {
        let pipeline: Pipeline<_, StandardScaler<f64, ArrayFeatureVector<f64, 1>, 1>, 0> =
            Pipeline::new(TestIndicators::new(10.0));

        assert!(approx_eq(pipeline.values()[0], 10.0));
    }

    #[test]
    fn dispatch_chains_transformers_and_exposes_last_output() {
        let mut pipeline: Pipeline<_, StandardScaler<f64, ArrayFeatureVector<f64, 1>, 1>, 2> =
            Pipeline::new(TestIndicators::new(10.0));
        pipeline
            .add_transformer(StandardScaler::new(
                [0],
                [0],
                2.0,
                2.0,
                ArrayFeatureVector::<f64, 1>::new(),
            ))
            .unwrap();
        pipeline
            .add_transformer(StandardScaler::new(
                [0],
                [0],
                1.0,
                2.0,
                ArrayFeatureVector::<f64, 1>::new(),
            ))
            .unwrap();

        pipeline.dispatch(&Event::price(crate::ticker::intern("TEST"), 0.0, 0));

        assert!(approx_eq(pipeline.values()[0], 1.5));
    }
}

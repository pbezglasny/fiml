use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::features::IndicatorFeatures;
use crate::features::transformers::Transformation;
use crate::{Event, FeatureVector, FimlError, Float};

pub struct Pipeline<I, T, F, V, const TRANSFORMER_SIZE: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: IndicatorFeatures<F = F>,
    T: Transformation<F = F, OutputVector = V>,
{
    indicators: I,
    transformers: [MaybeUninit<T>; TRANSFORMER_SIZE],
    num_transformers: usize,
    _phantom: PhantomData<V>,
}

impl<I, T, F, V, const NUM_TRANSFORMERS_SIZE: usize> Pipeline<I, T, F, V, NUM_TRANSFORMERS_SIZE>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: IndicatorFeatures<F = F>,
    T: Transformation<F = F, OutputVector = V>,
{
    pub fn new(indicators: I) -> Self {
        Self {
            indicators,
            transformers: [const { MaybeUninit::uninit() }; NUM_TRANSFORMERS_SIZE],
            num_transformers: 0,
            _phantom: PhantomData::<V>,
        }
    }

    pub fn add_transformer(&mut self, transformer: T) -> Result<(), FimlError> {
        if self.num_transformers < NUM_TRANSFORMERS_SIZE {
            self.transformers[self.num_transformers].write(transformer);
            self.num_transformers += 1;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                transformer_index = self.num_transformers - 1,
                transformer_count = self.num_transformers,
                transformer_capacity = NUM_TRANSFORMERS_SIZE,
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

    pub fn dispatch(&mut self, event: &Event<I::F>) {
        self.indicators.dispatch(event);
        if self.num_transformers == 0 {
            return;
        }

        let first = unsafe { self.transformers[0].assume_init_mut() };
        first.transform(self.indicators.feature_vector());

        for i in 1..self.num_transformers {
            let (previous, current) = self.transformers.split_at_mut(i);
            let prev_transformation = unsafe { previous[i - 1].assume_init_ref() };
            let current_transformation = unsafe { current[0].assume_init_mut() };
            current_transformation.transform(prev_transformation.output_values());
        }
    }

    pub fn values(&self) -> &[F] {
        if self.num_transformers == 0 {
            self.indicators.feature_vector().values()
        } else {
            let last = unsafe { self.transformers[self.num_transformers - 1].assume_init_ref() };
            last.output_values().values()
        }
    }
}

impl<I, T, F, V, const NUM_TRANSFORMERS_SIZE: usize> Drop
    for Pipeline<I, T, F, V, NUM_TRANSFORMERS_SIZE>
where
    F: Float,
    V: FeatureVector<F = F>,
    I: IndicatorFeatures<F = F>,
    T: Transformation<F = F, OutputVector = V>,
{
    fn drop(&mut self) {
        // SAFETY: the first `num_transformers` entries are initialized by
        // `add_transformer`.
        for slot in &mut self.transformers[..self.num_transformers] {
            unsafe { slot.assume_init_drop() };
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
        type F = f64;
        type FeatureVector = ArrayFeatureVector<f64, 1>;

        fn feature_vector(&self) -> &Self::FeatureVector {
            &self.cells
        }

        fn dispatch(&mut self, _event: &Event<Self::F>) {}

        fn index_of(&self, _ticker: Ticker, _name: &str) -> Option<usize> {
            None
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn values_returns_indicator_output_without_transformers() {
        let pipeline: Pipeline<
            _,
            StandardScaler<f64, ArrayFeatureVector<f64, 1>, 1>,
            f64,
            ArrayFeatureVector<f64, 1>,
            0,
        > = Pipeline::new(TestIndicators::new(10.0));

        assert!(approx_eq(pipeline.values()[0], 10.0));
    }

    #[test]
    fn dispatch_chains_transformers_and_exposes_last_output() {
        let mut pipeline: Pipeline<
            _,
            StandardScaler<f64, ArrayFeatureVector<f64, 1>, 1>,
            f64,
            ArrayFeatureVector<f64, 1>,
            2,
        > = Pipeline::new(TestIndicators::new(10.0));
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

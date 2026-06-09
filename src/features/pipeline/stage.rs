use crate::{FeatureVector, Float};

pub trait PipelineStage {
    type F: Float;
    type V: FeatureVector<F = Self::F>;
    fn trasform(&mut self, input: &Self::V);

    fn get_values(&self) -> &Self::V;
}

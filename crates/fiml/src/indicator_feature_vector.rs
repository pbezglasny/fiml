use crate::features::builtin::IndicatorFeaturesEnum;
use crate::{Event, EventKind, FeatureVector, FimlError, Float, Symbol};

use std::marker::PhantomData;
use std::mem::MaybeUninit;

struct SymbolRouter {}

impl SymbolRouter {}

struct EventRouter {
    symbol_to_index: Box<[Option<u16>]>,
}

impl EventRouter {
    fn route(&self, symbol: Symbol, event_kind: EventKind) {}
}

pub struct IndicatorFeatureVector<F, V, const M: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    feature_vector: V,
    features: [MaybeUninit<IndicatorFeaturesEnum<F>>; M],
    feature_count: usize,
    /// Indicies of timed features, will require to update time
    /// window at every update event
    timed_features: [usize; M],
    timed_feature_count: usize,

    last_timestamp: Option<i64>,
    _marker: PhantomData<F>,
}

pub struct UpdateResult {
    features_updated: usize,
}

impl<F, V, const M: usize> IndicatorFeatureVector<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub fn handle_event(&mut self, event: &Event<F>) -> Result<UpdateResult, FimlError> {
        // take features that subscribed on this type of event
        // call update of feature and store in vector
        todo!()
    }

    fn observe_timed_features(&self) {
        todo!()
    }
}

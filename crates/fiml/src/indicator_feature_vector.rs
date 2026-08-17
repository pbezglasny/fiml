use crate::features::builtin::IndicatorFeaturesEnum;
use crate::{EVENT_KIND_COUNT, Event, EventKind, FeatureVector, FimlError, Float, Symbol};

use std::marker::PhantomData;
use std::mem::MaybeUninit;
// Price,
//     Volume,
//     Trade,
//     OrderBook,
//     Time,

#[derive(Clone, Copy, Default)]
struct FeatureRange {
    start: u16,
    len: u16,
}

impl FeatureRange {
    fn as_slice(self, subscribers: &[u16]) -> &[u16] {
        let start = usize::from(self.start);
        let end = start + usize::from(self.len);
        &subscribers[start..end]
    }
}

struct SymbolRouter {
    price_subscribers: [FeatureRange; EVENT_KIND_COUNT],
}

struct EventRouter {
    symbol_to_index: Box<[Option<u16>]>,
    symbol_routers: Box<[SymbolRouter]>,
}

impl EventRouter {
    /// Return indecies of features that subscribed on this event type
    #[inline]
    fn route(&self, symbol: Symbol, event_kind: EventKind) -> &[u16] {
        todo!()
    }
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

    /// Remove expired data from timed features
    fn observe_timed_features(&self) {
        todo!()
    }
}

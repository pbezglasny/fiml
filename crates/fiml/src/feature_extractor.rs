use crate::features::builtin::IndicatorFeaturesEnum;
use crate::features::compiler::OutputSpan;
use crate::{EVENT_KIND_COUNT, Event, EventKind, FeatureVector, FimlError, Float, Symbol};

use std::marker::PhantomData;
use std::mem::MaybeUninit;

#[derive(Clone, Copy, Default)]
struct SubscriberRange {
    /// Index of the first entry in [`EventRouter::subscribers`].
    start: u16,
    /// Number of consecutive entries in [`EventRouter::subscribers`].
    len: u16,
}

impl SubscriberRange {
    fn as_slice(self, subscribers: &[u16]) -> &[u16] {
        let start = usize::from(self.start);
        let end = start + usize::from(self.len);
        &subscribers[start..end]
    }
}

struct SymbolRouter {
    /// Subscriber ranges for this symbol.
    ///
    /// Each array index is an [`EventKind`] converted to `usize`. Its value is
    /// the range in [`EventRouter::subscribers`] containing the runtime feature
    /// indices subscribed to that symbol and event kind.
    event_subscribers: [SubscriberRange; EVENT_KIND_COUNT],
}

/// Maps an event's symbol and kind to the runtime features that consume it.
struct EventRouter {
    /// Maps interned symbols to their symbol-specific routers.
    ///
    /// Each array index is [`Symbol::index`]. Its value is either the index of
    /// that symbol's [`SymbolRouter`] in [`Self::symbol_routers`] or `None` when
    /// no feature subscribes to the symbol.
    symbol_to_index: Box<[Option<u16>]>,
    /// Symbol-specific event routing tables.
    ///
    /// Each array index is the compact router index stored in
    /// [`Self::symbol_to_index`]. Its value contains all event-kind subscriber
    /// ranges configured for that symbol.
    symbol_routers: Box<[SymbolRouter]>,
    /// Flattened runtime feature indices referenced by subscriber ranges.
    ///
    /// Each array index is a flattened subscriber position addressed by a
    /// [`SubscriberRange`]. Its value is an index into
    /// [`FeatureExtractor::features`]. Entries belonging to one route are
    /// stored contiguously.
    subscribers: Box<[u16]>,
    /// Range in [`Self::subscribers`] containing the runtime feature indices
    /// invoked for every accepted event, including timed features.
    always_subscribers: SubscriberRange,
}

impl EventRouter {
    pub(crate) fn new(
        symbol_to_index: Box<[Option<u16>]>,
        symbol_routers: Box<[SymbolRouter]>,
        subscribers: Box<[u16]>,
        always_subscribers: SubscriberRange,
    ) -> Self {
        Self {
            symbol_to_index,
            symbol_routers,
            subscribers,
            always_subscribers,
        }
    }

    /// Returns runtime feature indices subscribed to this symbol and event kind.
    #[inline]
    fn route(&self, symbol: Symbol, event_kind: EventKind) -> &[u16] {
        let Some(symbol_router_index) = self.symbol_to_index.get(symbol.index()).copied().flatten()
        else {
            return &[];
        };

        let symbol_router = &self.symbol_routers[usize::from(symbol_router_index)];
        symbol_router.event_subscribers[event_kind as usize].as_slice(&self.subscribers)
    }

    /// Returns runtime feature indices invoked for every accepted event.
    #[inline]
    fn always(&self) -> &[u16] {
        self.always_subscribers.as_slice(&self.subscribers)
    }
}

/// Stateful, fixed-capacity extractor that routes events to subscribed features.
///
/// The extractor owns the runtime feature state and the output feature vector.
/// Handling an event updates the subscribed features directly in that vector
/// without allocating on the event-processing path.
pub struct FeatureExtractor<F, V, const M: usize>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    feature_vector: V,
    /// Runtime features stored in the initialized `[..feature_count]` prefix.
    features: [MaybeUninit<IndicatorFeaturesEnum<F>>; M],
    /// Output spans corresponding one-to-one with [`Self::features`].
    ///
    /// Each initialized array index is a runtime feature index. Its value is
    /// the contiguous range of feature-vector cells written by the feature at
    /// the same index in [`Self::features`].
    output_spans: [MaybeUninit<OutputSpan>; M],
    feature_count: usize,
    event_router: EventRouter,

    last_timestamp: Option<i64>,
    _marker: PhantomData<F>,
}

pub struct UpdateResult {
    /// Number of runtime feature handlers invoked for the accepted event.
    pub features_updated: usize,
}

impl<F, V, const M: usize> FeatureExtractor<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub fn handle_event(&mut self, event: &Event<F>) -> Result<UpdateResult, FimlError> {
        if let Some(previous_timestamp) = self.last_timestamp
            && previous_timestamp > event.timestamp()
        {
            return Err(FimlError::TimestampOutOfOrder {
                symbol: event.symbol(),
                event_kind: event.kind(),
                timestamp: event.timestamp(),
                previous_timestamp,
            });
        }

        let subscribed_features = self.event_router.route(event.symbol(), event.kind());
        Self::update_subscribers(
            &mut self.features,
            &self.output_spans,
            &mut self.feature_vector,
            subscribed_features,
            event,
        );

        let always_features = self.event_router.always();
        Self::update_subscribers(
            &mut self.features,
            &self.output_spans,
            &mut self.feature_vector,
            always_features,
            event,
        );

        self.last_timestamp = Some(event.timestamp());
        Ok(UpdateResult {
            features_updated: subscribed_features.len() + always_features.len(),
        })
    }

    /// Return timestamp of last seen event
    pub fn last_timestamp(&self) -> Option<i64> {
        self.last_timestamp
    }

    /// Return feature vector
    pub fn feature_vector(&self) -> &V {
        &self.feature_vector
    }

    fn update_subscribers(
        features: &mut [MaybeUninit<IndicatorFeaturesEnum<F>>; M],
        output_spans: &[MaybeUninit<OutputSpan>; M],
        feature_vector: &mut V,
        subscribers: &[u16],
        event: &Event<F>,
    ) {
        for &feature_index in subscribers {
            let feature_index = usize::from(feature_index);

            // SAFETY: construction initializes matching entries in `features`
            // and `output_spans` before registering the index with the router.
            let feature = unsafe { features[feature_index].assume_init_mut() };
            let output_span = unsafe { *output_spans[feature_index].assume_init_ref() };
            feature.update(event, output_span, feature_vector);
        }
    }
}

impl<F, V, const M: usize> Drop for FeatureExtractor<F, V, M>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    fn drop(&mut self) {
        // SAFETY: construction initializes exactly the
        // `features[..feature_count]` prefix.
        for feature in &mut self.features[..self.feature_count] {
            unsafe { feature.assume_init_drop() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector};

    #[test]
    fn routes_runtime_feature_indices_by_symbol_and_event_kind() {
        let btc = Symbol::new("router-btc");
        let eth = Symbol::new("router-eth");
        let symbol_count = btc.index().max(eth.index()) + 1;

        let mut symbol_to_index = vec![None; symbol_count];
        symbol_to_index[btc.index()] = Some(0);
        symbol_to_index[eth.index()] = Some(1);

        let mut btc_subscribers = [SubscriberRange::default(); EVENT_KIND_COUNT];
        btc_subscribers[EventKind::Trade as usize] = SubscriberRange { start: 0, len: 2 };

        let mut eth_subscribers = [SubscriberRange::default(); EVENT_KIND_COUNT];
        eth_subscribers[EventKind::Price as usize] = SubscriberRange { start: 2, len: 1 };

        let router = EventRouter {
            symbol_to_index: symbol_to_index.into_boxed_slice(),
            symbol_routers: vec![
                SymbolRouter {
                    event_subscribers: btc_subscribers,
                },
                SymbolRouter {
                    event_subscribers: eth_subscribers,
                },
            ]
            .into_boxed_slice(),
            subscribers: vec![3, 7, 4, 8, 9].into_boxed_slice(),
            always_subscribers: SubscriberRange { start: 3, len: 2 },
        };

        assert_eq!(router.route(btc, EventKind::Trade), [3, 7]);
        assert_eq!(router.route(eth, EventKind::Price), [4]);
        assert!(router.route(btc, EventKind::Price).is_empty());
        assert_eq!(router.always(), [8, 9]);
    }

    #[test]
    fn returns_empty_slice_for_unmapped_symbol() {
        let configured = Symbol::new("router-configured");
        let mut symbol_to_index = vec![None; configured.index() + 1];
        symbol_to_index[configured.index()] = Some(0);

        let router = EventRouter {
            symbol_to_index: symbol_to_index.into_boxed_slice(),
            symbol_routers: vec![SymbolRouter {
                event_subscribers: [SubscriberRange::default(); EVENT_KIND_COUNT],
            }]
            .into_boxed_slice(),
            subscribers: Box::new([]),
            always_subscribers: SubscriberRange::default(),
        };

        assert!(
            router
                .route(Symbol::new("router-unmapped"), EventKind::Trade)
                .is_empty()
        );
    }

    #[test]
    fn passes_the_matching_output_span_to_the_feature() {
        let mut features = [const { MaybeUninit::uninit() }; 1];
        features[0].write(crate::features::builtin::day_of_week::build());

        let mut output_spans = [const { MaybeUninit::uninit() }; 1];
        output_spans[0].write(OutputSpan { start: 1, count: 1 });

        let mut vector = FeatureExtractor::<f64, ArrayFeatureVector<f64, 2>, 1> {
            feature_vector: ArrayFeatureVector::new(),
            features,
            output_spans,
            feature_count: 1,
            event_router: EventRouter {
                symbol_to_index: Box::new([]),
                symbol_routers: Box::new([]),
                subscribers: vec![0].into_boxed_slice(),
                always_subscribers: SubscriberRange { start: 0, len: 1 },
            },
            last_timestamp: None,
            _marker: PhantomData,
        };

        let result = vector
            .handle_event(&Event::time(1_609_459_200_000))
            .unwrap();

        assert_eq!(result.features_updated, 1);
        assert_eq!(vector.feature_vector().values(), [0.0, 5.0]);
    }
}

use crate::features::FeatureRoute;
use crate::features::compiler::{Compilation, OutputSpan};
use crate::features::derivation::FeatureDerivation;
use crate::features::feature_extractor_builder::FeatureExtractorBuilder;
use crate::{
    EVENT_KIND_COUNT, Event, EventKind, FeatureId, FeatureVector, FimlError, Float, Result, Symbol,
};

#[derive(Clone, Copy, Default)]
pub(crate) struct SubscriberRange {
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

pub(crate) struct SymbolRouter {
    /// Subscriber ranges for this symbol.
    ///
    /// Each array index is an [`EventKind`] converted to `usize`. Its value is
    /// the range in [`EventRouter::subscribers`] containing the runtime feature
    /// indices subscribed to that symbol and event kind.
    event_subscribers: [SubscriberRange; EVENT_KIND_COUNT],
}

/// Maps an event's symbol and kind to the runtime features that consume it.
pub(crate) struct EventRouter {
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
    pub(crate) fn from_routes(routes: &[(Symbol, FeatureRoute)]) -> Result<Self> {
        let max_symbol_index = routes
            .iter()
            .filter_map(|(symbol, route)| match route {
                FeatureRoute::Kind(_) => Some(symbol.index()),
                FeatureRoute::Every => None,
            })
            .max();
        let mut symbol_to_index = vec![None; max_symbol_index.map_or(0, |index| index + 1)];
        let mut grouped_subscribers: Vec<[Vec<u16>; EVENT_KIND_COUNT]> = Vec::new();
        let mut always_subscribers = Vec::new();

        for (feature_index, &(symbol, route)) in routes.iter().enumerate() {
            let feature_index = u16::try_from(feature_index).map_err(|_| {
                FimlError::InvalidArgument(format!(
                    "runtime feature count exceeds router limit of {}",
                    u16::MAX
                ))
            })?;

            match route {
                FeatureRoute::Every => always_subscribers.push(feature_index),
                FeatureRoute::Kind(event_kind) => {
                    let symbol_index = symbol.index();
                    let router_index = match symbol_to_index[symbol_index] {
                        Some(router_index) => usize::from(router_index),
                        None => {
                            let router_index =
                                u16::try_from(grouped_subscribers.len()).map_err(|_| {
                                    FimlError::InvalidArgument(format!(
                                        "symbol router count exceeds limit of {}",
                                        u16::MAX
                                    ))
                                })?;
                            symbol_to_index[symbol_index] = Some(router_index);
                            grouped_subscribers.push(std::array::from_fn(|_| Vec::<u16>::new()));
                            usize::from(router_index)
                        }
                    };
                    grouped_subscribers[router_index][event_kind as usize].push(feature_index);
                }
            }
        }

        let mut subscribers = Vec::with_capacity(routes.len());
        let mut symbol_routers = Vec::with_capacity(grouped_subscribers.len());
        for event_subscriber_groups in grouped_subscribers {
            let mut event_subscribers = [SubscriberRange::default(); EVENT_KIND_COUNT];
            for (event_kind_index, group) in event_subscriber_groups.into_iter().enumerate() {
                event_subscribers[event_kind_index] =
                    Self::append_subscribers(&mut subscribers, group)?;
            }
            symbol_routers.push(SymbolRouter { event_subscribers });
        }

        let always_subscribers = Self::append_subscribers(&mut subscribers, always_subscribers)?;

        Ok(Self {
            symbol_to_index: symbol_to_index.into_boxed_slice(),
            symbol_routers: symbol_routers.into_boxed_slice(),
            subscribers: subscribers.into_boxed_slice(),
            always_subscribers,
        })
    }

    fn append_subscribers(subscribers: &mut Vec<u16>, group: Vec<u16>) -> Result<SubscriberRange> {
        let start = u16::try_from(subscribers.len()).map_err(|_| {
            FimlError::InvalidArgument(format!(
                "subscriber count exceeds router limit of {}",
                u16::MAX
            ))
        })?;
        let len = u16::try_from(group.len()).map_err(|_| {
            FimlError::InvalidArgument(format!(
                "subscriber group exceeds router limit of {}",
                u16::MAX
            ))
        })?;
        subscribers.extend(group);
        Ok(SubscriberRange { start, len })
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

/// Stateful extractor that routes events to subscribed features.
///
/// The extractor owns the runtime feature state and the output feature vector.
/// Handling an event updates the subscribed features directly in that vector
/// without allocating on the event-processing path.
pub struct FeatureExtractor<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    feature_vector: V,
    /// Runtime features indexed by the event router.
    features: Box<[FeatureDerivation<F>]>,
    /// Output spans corresponding one-to-one with [`Self::features`].
    ///
    /// Each slice index is a runtime feature index. Its value is the contiguous
    /// range of feature-vector cells written by the feature at the same index
    /// in [`Self::features`].
    output_spans: Box<[OutputSpan]>,
    /// User-facing IDs in feature-vector index order.
    feature_ids: Box<[FeatureId]>,
    event_router: EventRouter,

    last_timestamp: Option<i64>,
}

pub struct UpdateResult {
    /// Number of runtime feature handlers invoked for the accepted event.
    pub features_updated: usize,
}

impl<F, V> FeatureExtractor<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new(feature_vector: V, compilation: Compilation<F>) -> Self {
        debug_assert_eq!(compilation.features.len(), compilation.output_spans.len());

        Self {
            feature_vector,
            features: compilation.features,
            output_spans: compilation.output_spans,
            feature_ids: compilation.feature_ids,
            event_router: compilation.event_router,
            last_timestamp: None,
        }
    }

    pub fn builder(output_vector: V) -> FeatureExtractorBuilder<F, V> {
        FeatureExtractorBuilder::new(output_vector)
    }

    pub fn handle_event(&mut self, event: &Event<F>) -> Result<UpdateResult> {
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

    /// Return feature IDs in output-vector index order.
    pub fn feature_ids(&self) -> &[FeatureId] {
        &self.feature_ids
    }

    /// Resolve a feature ID to its output-vector index.
    pub fn feature_index(&self, feature_id: &FeatureId) -> Option<usize> {
        self.feature_ids.iter().position(|id| id == feature_id)
    }

    fn update_subscribers(
        features: &mut [FeatureDerivation<F>],
        output_spans: &[OutputSpan],
        feature_vector: &mut V,
        subscribers: &[u16],
        event: &Event<F>,
    ) {
        for &feature_index in subscribers {
            let feature_index = usize::from(feature_index);

            features[feature_index].update(event, output_spans[feature_index], feature_vector);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArrayFeatureVector, EventField, FeatureDefinition, FeatureKey, FeatureSource,
        FeatureVector, WarmupPolicy,
    };

    #[test]
    fn builder_infers_types_from_output_vector() {
        let _builder = FeatureExtractor::builder(ArrayFeatureVector::<f64, 2>::new());
    }

    #[test]
    fn builder_compiles_definitions_and_routes_events() {
        let symbol = Symbol::new("extractor-builder");
        let first_key = FeatureKey::Sma {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            window: 1,
            warmup_policy: WarmupPolicy::FullWindow,
        };
        let second_key = FeatureKey::Sma {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            window: 2,
            warmup_policy: WarmupPolicy::FullWindow,
        };
        let first_id = FeatureId::from(&first_key);
        let second_id = FeatureId::from(&second_key);
        let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<f64, 2>::new())
            .add_feature(FeatureDefinition::with_default_id(first_key))
            .add_feature(FeatureDefinition::with_default_id(second_key))
            .build()
            .unwrap();

        assert_eq!(extractor.feature_ids(), [first_id.clone(), second_id]);
        assert_eq!(extractor.feature_index(&first_id), Some(0));
        assert_eq!(
            extractor
                .handle_event(&Event::volume(symbol, 100.0, 0))
                .unwrap()
                .features_updated,
            0
        );
        assert_eq!(
            extractor
                .handle_event(&Event::price(symbol, 10.0, 1))
                .unwrap()
                .features_updated,
            1
        );
        extractor
            .handle_event(&Event::price(symbol, 20.0, 2))
            .unwrap();

        assert_eq!(extractor.feature_vector().values(), [20.0, 15.0]);
    }

    #[test]
    fn builder_validates_output_vector_length() {
        let key = FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::EveryEvent,
        };

        let result = FeatureExtractor::builder(ArrayFeatureVector::<f64, 2>::new())
            .add_feature(FeatureDefinition::with_default_id(key))
            .build();

        assert!(matches!(result, Err(FimlError::OutputCountMismatch { .. })));
    }

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
        let features = vec![crate::features::derivation::day_of_week::build()].into_boxed_slice();
        let output_spans = vec![OutputSpan { start: 1, count: 1 }].into_boxed_slice();

        let mut vector = FeatureExtractor::<f64, ArrayFeatureVector<f64, 2>> {
            feature_vector: ArrayFeatureVector::new(),
            features,
            output_spans,
            feature_ids: vec![FeatureId::new("day")].into_boxed_slice(),
            event_router: EventRouter {
                symbol_to_index: Box::new([]),
                symbol_routers: Box::new([]),
                subscribers: vec![0].into_boxed_slice(),
                always_subscribers: SubscriberRange { start: 0, len: 1 },
            },
            last_timestamp: None,
        };

        let result = vector
            .handle_event(&Event::time(1_609_459_200_000))
            .unwrap();

        assert_eq!(result.features_updated, 1);
        assert_eq!(vector.feature_vector().values(), [0.0, 5.0]);
    }
}

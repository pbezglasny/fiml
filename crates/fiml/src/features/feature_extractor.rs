use crate::features::FeatureRoute;
use crate::features::compiler::{Compilation, OutputSpan};
use crate::features::derivation::FeatureDerivation;
use crate::features::feature_extractor_builder::FeatureExtractorBuilder;
use crate::order_book::{
    OrderBook, OrderBookUpdate, OrderBookUpdateOutcome, OrderBookUpdateRef, PreparedOrderBookUpdate,
};
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
    /// Features that consume the visible order-book state after it changes.
    order_book_subscribers: SubscriberRange,
}

struct PendingSymbolSubscribers {
    event_subscribers: [Vec<u16>; EVENT_KIND_COUNT],
    order_book_subscribers: Vec<u16>,
}

impl PendingSymbolSubscribers {
    fn new() -> Self {
        Self {
            event_subscribers: std::array::from_fn(|_| Vec::new()),
            order_book_subscribers: Vec::new(),
        }
    }
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
    /// invoked for any accepted event, including timed features.
    any_subscribers: SubscriberRange,
}

impl EventRouter {
    pub(crate) fn from_routes(routes: &[(Symbol, FeatureRoute)]) -> Result<Self> {
        let max_symbol_index = routes
            .iter()
            .filter_map(|(symbol, route)| match route {
                FeatureRoute::Kind(_) | FeatureRoute::OrderBook => Some(symbol.index()),
                FeatureRoute::Any => None,
            })
            .max();
        let mut symbol_to_index = vec![None; max_symbol_index.map_or(0, |index| index + 1)];
        let mut grouped_subscribers = Vec::<PendingSymbolSubscribers>::new();
        let mut any_subscribers = Vec::new();

        for (feature_index, &(symbol, route)) in routes.iter().enumerate() {
            let feature_index = u16::try_from(feature_index).map_err(|_| {
                FimlError::InvalidArgument(format!(
                    "runtime feature count exceeds router limit of {}",
                    u16::MAX
                ))
            })?;

            match route {
                FeatureRoute::Any => any_subscribers.push(feature_index),
                route @ (FeatureRoute::Kind(_) | FeatureRoute::OrderBook) => {
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
                            grouped_subscribers.push(PendingSymbolSubscribers::new());
                            usize::from(router_index)
                        }
                    };
                    match route {
                        FeatureRoute::Kind(event_kind) => grouped_subscribers[router_index]
                            .event_subscribers[event_kind as usize]
                            .push(feature_index),
                        FeatureRoute::OrderBook => grouped_subscribers[router_index]
                            .order_book_subscribers
                            .push(feature_index),
                        FeatureRoute::Any => unreachable!("handled before symbol routing"),
                    }
                }
            }
        }

        let mut subscribers = Vec::with_capacity(routes.len());
        let mut symbol_routers = Vec::with_capacity(grouped_subscribers.len());
        for grouped in grouped_subscribers {
            let mut event_subscribers = [SubscriberRange::default(); EVENT_KIND_COUNT];
            for (event_kind_index, group) in grouped.event_subscribers.into_iter().enumerate() {
                event_subscribers[event_kind_index] =
                    Self::append_subscribers(&mut subscribers, group)?;
            }
            let order_book_subscribers =
                Self::append_subscribers(&mut subscribers, grouped.order_book_subscribers)?;
            symbol_routers.push(SymbolRouter {
                event_subscribers,
                order_book_subscribers,
            });
        }

        let any_subscribers = Self::append_subscribers(&mut subscribers, any_subscribers)?;

        Ok(Self {
            symbol_to_index: symbol_to_index.into_boxed_slice(),
            symbol_routers: symbol_routers.into_boxed_slice(),
            subscribers: subscribers.into_boxed_slice(),
            any_subscribers,
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

    /// Returns runtime feature indices invoked for any accepted event.
    #[inline]
    fn any(&self) -> &[u16] {
        self.any_subscribers.as_slice(&self.subscribers)
    }

    /// Returns runtime feature indices subscribed to this symbol's order book.
    #[inline]
    fn order_book(&self, symbol: Symbol) -> &[u16] {
        let Some(symbol_router_index) = self.symbol_to_index.get(symbol.index()).copied().flatten()
        else {
            return &[];
        };

        let symbol_router = &self.symbol_routers[usize::from(symbol_router_index)];
        symbol_router
            .order_book_subscribers
            .as_slice(&self.subscribers)
    }
}

/// Dense symbol-indexed storage for caller-configured order books.
struct OrderBookStorage {
    symbol_to_index: Box<[Option<u16>]>,
    books: Box<[OrderBook]>,
}

impl OrderBookStorage {
    fn new(configured: Vec<(Symbol, OrderBook)>) -> Result<Self> {
        let max_symbol_index = configured.iter().map(|(symbol, _)| symbol.index()).max();
        let mut symbol_to_index = vec![None; max_symbol_index.map_or(0, |index| index + 1)];
        let mut books = Vec::with_capacity(configured.len());

        for (symbol, book) in configured {
            if symbol_to_index[symbol.index()].is_some() {
                return Err(FimlError::DuplicateOrderBook { symbol });
            }
            let book_index = u16::try_from(books.len()).map_err(|_| {
                FimlError::InvalidArgument(format!(
                    "order-book count exceeds limit of {}",
                    u16::MAX
                ))
            })?;
            symbol_to_index[symbol.index()] = Some(book_index);
            books.push(book);
        }

        Ok(Self {
            symbol_to_index: symbol_to_index.into_boxed_slice(),
            books: books.into_boxed_slice(),
        })
    }

    fn get(&self, symbol: Symbol) -> Option<&OrderBook> {
        self.index(symbol).and_then(|index| self.books.get(index))
    }

    fn prepare_update(
        &self,
        symbol: Symbol,
        update: OrderBookUpdateRef<'_>,
    ) -> Result<PreparedOrderBookUpdate> {
        let index = self
            .index(symbol)
            .ok_or(FimlError::OrderBookNotConfigured { symbol })?;
        Ok(self.books[index].prepare_update(update))
    }

    fn commit_update(
        &mut self,
        symbol: Symbol,
        prepared: PreparedOrderBookUpdate,
        update: OrderBookUpdate,
    ) -> Result<OrderBookUpdateOutcome> {
        let index = self
            .index(symbol)
            .ok_or(FimlError::OrderBookNotConfigured { symbol })?;
        self.books[index]
            .commit_update(prepared, update)
            .map_err(Into::into)
    }

    fn index(&self, symbol: Symbol) -> Option<usize> {
        self.symbol_to_index
            .get(symbol.index())
            .copied()
            .flatten()
            .map(usize::from)
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
    order_books: OrderBookStorage,
    last_timestamp: Option<i64>,
}

#[derive(Clone, Copy)]
pub struct UpdateResult {
    /// Number of runtime feature handlers invoked for the accepted event.
    pub features_updated: usize,
}

impl UpdateResult {
    pub fn combine_with(self, other: UpdateResult) -> Self {
        Self {
            features_updated: self.features_updated + other.features_updated,
        }
    }
}

impl<F, V> FeatureExtractor<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new(
        feature_vector: V,
        compilation: Compilation<F>,
        configured_order_books: Vec<(Symbol, OrderBook)>,
    ) -> Result<Self> {
        debug_assert_eq!(compilation.features.len(), compilation.output_spans.len());

        let order_books = OrderBookStorage::new(configured_order_books)?;

        Ok(Self {
            feature_vector,
            features: compilation.features,
            output_spans: compilation.output_spans,
            feature_ids: compilation.feature_ids,
            event_router: compilation.event_router,
            order_books,
            last_timestamp: None,
        })
    }

    pub fn builder(output_vector: V) -> FeatureExtractorBuilder<F, V> {
        FeatureExtractorBuilder::new(output_vector)
    }

    fn prepare_order_book_event(
        &self,
        symbol: Symbol,
        update: OrderBookUpdateRef<'_>,
    ) -> Result<Option<PreparedOrderBookUpdate>> {
        let subscribers = self.event_router.order_book(symbol);
        if self.order_books.get(symbol).is_none() {
            if subscribers.is_empty() {
                return Ok(None);
            }
            return Err(FimlError::OrderBookNotConfigured { symbol });
        }

        self.order_books.prepare_update(symbol, update).map(Some)
    }

    fn commit_order_book_event(
        &mut self,
        symbol: Symbol,
        timestamp: i64,
        prepared: PreparedOrderBookUpdate,
        update: OrderBookUpdate,
    ) -> Result<UpdateResult> {
        let subscribers = self.event_router.order_book(symbol);
        let outcome = self.order_books.commit_update(symbol, prepared, update)?;

        if !matches!(
            outcome,
            OrderBookUpdateOutcome::Applied | OrderBookUpdateOutcome::Resynchronized
        ) {
            return Ok(UpdateResult {
                features_updated: 0,
            });
        }

        let order_book = self
            .order_books
            .get(symbol)
            .expect("the order book accepted an update and must still be configured");
        let features_updated = Self::update_order_book_subscribers(
            &mut self.features,
            &self.output_spans,
            &mut self.feature_vector,
            subscribers,
            order_book,
            timestamp,
        );

        Ok(UpdateResult { features_updated })
    }

    fn update_any_features(&mut self, event: &Event<F>) -> UpdateResult {
        let any_features = self.event_router.any();
        Self::update_subscribers(
            &mut self.features,
            &self.output_spans,
            &mut self.feature_vector,
            any_features,
            event,
        );
        UpdateResult {
            features_updated: any_features.len(),
        }
    }

    fn update_event_features(&mut self, event: &Event<F>) -> UpdateResult {
        let subscribed_features = self.event_router.route(event.symbol(), event.kind());
        Self::update_subscribers(
            &mut self.features,
            &self.output_spans,
            &mut self.feature_vector,
            subscribed_features,
            event,
        );

        UpdateResult {
            features_updated: subscribed_features.len(),
        }
    }

    pub fn handle_event(&mut self, event: Event<F>) -> Result<UpdateResult> {
        // TODO: use separate counters for each symbol
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

        let symbol = event.symbol();
        let timestamp = event.timestamp();
        let prepared_order_book_update = event
            .order_book_update()
            .map(|update| self.prepare_order_book_event(symbol, update))
            .transpose()?
            .flatten();

        if prepared_order_book_update
            .as_ref()
            .is_some_and(PreparedOrderBookUpdate::is_rejected)
        {
            let update = event
                .into_order_book_update()
                .expect("a prepared order-book update must come from an order-book event");
            let prepared = prepared_order_book_update
                .expect("the rejected order-book update was prepared above");
            let result = self.order_books.commit_update(symbol, prepared, update);
            return match result {
                Err(error) => Err(error),
                Ok(_) => unreachable!("a rejected order-book update cannot commit successfully"),
            };
        }

        let any_features_result = self.update_any_features(&event);
        let event_features_result = self.update_event_features(&event);

        let order_book_features_result =
            match (prepared_order_book_update, event.into_order_book_update()) {
                (Some(prepared), Some(update)) => {
                    self.commit_order_book_event(symbol, timestamp, prepared, update)?
                }
                (None, _) => UpdateResult {
                    features_updated: 0,
                },
                (Some(_), None) => {
                    unreachable!("a prepared order-book update must come from an order-book event")
                }
            };
        self.last_timestamp = Some(timestamp);
        let total_updated = any_features_result
            .combine_with(event_features_result)
            .combine_with(order_book_features_result);
        Ok(total_updated)
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

    /// Return order book of given symbol
    pub fn order_book_of_symbol(&self, symbol: Symbol) -> Option<&OrderBook> {
        self.order_books.get(symbol)
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

    fn update_order_book_subscribers(
        features: &mut [FeatureDerivation<F>],
        output_spans: &[OutputSpan],
        feature_vector: &mut V,
        subscribers: &[u16],
        order_book: &OrderBook,
        timestamp: i64,
    ) -> usize {
        subscribers
            .iter()
            .copied()
            .filter(|&feature_index| {
                let feature_index = usize::from(feature_index);
                features[feature_index].update_order_book(
                    order_book,
                    timestamp,
                    output_spans[feature_index],
                    feature_vector,
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArrayFeatureVector, EventField, FeatureDefinition, FeatureKey, FeatureSource,
        FeatureVector, WarmupPolicy,
        order_book::{
            OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate, OrderBookSnapshot,
            OrderBookUpdateError, Side, UpdatePolicy,
        },
    };
    use rust_decimal::dec;

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
                .handle_event(Event::volume(symbol, 100.0, 0))
                .unwrap()
                .features_updated,
            0
        );
        assert_eq!(
            extractor
                .handle_event(Event::price(symbol, 10.0, 1))
                .unwrap()
                .features_updated,
            1
        );
        extractor
            .handle_event(Event::price(symbol, 20.0, 2))
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
                    order_book_subscribers: SubscriberRange::default(),
                },
                SymbolRouter {
                    event_subscribers: eth_subscribers,
                    order_book_subscribers: SubscriberRange::default(),
                },
            ]
            .into_boxed_slice(),
            subscribers: vec![3, 7, 4, 8, 9].into_boxed_slice(),
            any_subscribers: SubscriberRange { start: 3, len: 2 },
        };

        assert_eq!(router.route(btc, EventKind::Trade), [3, 7]);
        assert_eq!(router.route(eth, EventKind::Price), [4]);
        assert!(router.route(btc, EventKind::Price).is_empty());
        assert_eq!(router.any(), [8, 9]);
    }

    #[test]
    fn routes_order_book_features_by_symbol() {
        let btc = Symbol::new("router-book-btc");
        let eth = Symbol::new("router-book-eth");
        let router = EventRouter::from_routes(&[
            (btc, FeatureRoute::OrderBook),
            (eth, FeatureRoute::Kind(EventKind::Trade)),
            (btc, FeatureRoute::OrderBook),
            (Symbol::GLOBAL, FeatureRoute::Any),
        ])
        .unwrap();

        assert_eq!(router.order_book(btc), [0, 2]);
        assert!(router.order_book(eth).is_empty());
        assert_eq!(router.route(eth, EventKind::Trade), [1]);
        assert_eq!(router.any(), [3]);
    }

    #[test]
    fn builder_configures_and_updates_order_book_by_symbol() {
        let symbol = Symbol::new("extractor-order-book");
        let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<f64, 0>::new())
            .add_order_book(symbol, OrderBook::new(UpdatePolicy::Contiguous, 4))
            .build()
            .unwrap();

        let result = extractor
            .handle_event(Event::order_book_snapshot(
                symbol,
                10,
                OrderBookSnapshot::new(
                    7,
                    vec![OrderBookLevel::new(dec!(100), dec!(2))],
                    vec![OrderBookLevel::new(dec!(102), dec!(3))],
                ),
            ))
            .unwrap();

        assert_eq!(result.features_updated, 0);
        assert_eq!(
            extractor.order_book_of_symbol(symbol).unwrap().mid_price(),
            Some(dec!(101))
        );
        assert_eq!(extractor.last_timestamp(), Some(10));
    }

    #[test]
    fn rejected_order_book_event_does_not_advance_features_or_timestamp() {
        let symbol = Symbol::new("rejected-extractor-order-book");
        let any_event = FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::EveryEvent,
        };
        let raw_delta = FeatureKey::DayOfWeek {
            symbol,
            source: FeatureSource::Event(EventKind::OrderBookDelta),
        };
        let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<f64, 2>::new())
            .add_feature(FeatureDefinition::with_default_id(any_event))
            .add_feature(FeatureDefinition::with_default_id(raw_delta))
            .add_order_book(symbol, OrderBook::new(UpdatePolicy::Contiguous, 4))
            .build()
            .unwrap();

        extractor
            .handle_event(Event::order_book_snapshot(
                symbol,
                0,
                OrderBookSnapshot::new(0, Vec::new(), Vec::new()),
            ))
            .unwrap();
        extractor
            .handle_event(Event::order_book_delta(
                symbol,
                1,
                OrderBookDelta::new(1, Vec::new()),
            ))
            .unwrap();
        assert_eq!(extractor.feature_vector().values(), [4.0, 4.0]);

        let result = extractor.handle_event(Event::order_book_delta(
            symbol,
            86_400_000,
            OrderBookDelta::new(
                2,
                vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(-1))],
            ),
        ));

        assert!(matches!(
            result,
            Err(FimlError::OrderBookUpdateError {
                reason: OrderBookUpdateError::InvalidUpdate { .. }
            })
        ));
        assert_eq!(extractor.feature_vector().values(), [4.0, 4.0]);
        assert_eq!(extractor.last_timestamp(), Some(1));
        assert!(extractor.handle_event(Event::time(2)).is_ok());
    }

    #[test]
    fn rejected_sequence_gap_is_buffered_without_advancing_features() {
        let symbol = Symbol::new("sequence-gap-extractor-order-book");
        let any_event = FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::EveryEvent,
        };
        let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<f64, 1>::new())
            .add_feature(FeatureDefinition::with_default_id(any_event))
            .add_order_book(symbol, OrderBook::new(UpdatePolicy::Contiguous, 4))
            .build()
            .unwrap();

        extractor
            .handle_event(Event::order_book_snapshot(
                symbol,
                0,
                OrderBookSnapshot::new(100, Vec::new(), Vec::new()),
            ))
            .unwrap();
        let result = extractor.handle_event(Event::order_book_delta(
            symbol,
            86_400_000,
            OrderBookDelta::new(102, Vec::new()),
        ));

        assert!(matches!(
            result,
            Err(FimlError::OrderBookUpdateError {
                reason: OrderBookUpdateError::SequenceGap {
                    expected: 101,
                    received: 102,
                }
            })
        ));
        assert_eq!(extractor.feature_vector().values(), [4.0]);
        assert_eq!(extractor.last_timestamp(), Some(0));

        extractor
            .handle_event(Event::order_book_snapshot(
                symbol,
                1,
                OrderBookSnapshot::new(101, Vec::new(), Vec::new()),
            ))
            .unwrap();
        assert_eq!(
            extractor
                .order_book_of_symbol(symbol)
                .unwrap()
                .last_update_id(),
            Some(102)
        );
    }

    #[test]
    fn raw_order_book_event_features_do_not_require_book_state() {
        let symbol = Symbol::new("raw-order-book-event");
        let key = FeatureKey::DayOfWeek {
            symbol,
            source: FeatureSource::Event(EventKind::OrderBookDelta),
        };
        let mut extractor = FeatureExtractor::builder(ArrayFeatureVector::<f64, 1>::new())
            .add_feature(FeatureDefinition::with_default_id(key))
            .build()
            .unwrap();

        let result = extractor
            .handle_event(Event::order_book_delta(
                symbol,
                0,
                OrderBookDelta::new(1, Vec::new()),
            ))
            .unwrap();

        assert_eq!(result.features_updated, 1);
        assert!(!extractor.feature_vector().values()[0].is_nan());
        assert!(extractor.order_book_of_symbol(symbol).is_none());
    }

    #[test]
    fn builder_rejects_duplicate_order_books() {
        let symbol = Symbol::new("duplicate-extractor-order-book");
        let result = FeatureExtractor::builder(ArrayFeatureVector::<f64, 0>::new())
            .add_order_book(symbol, OrderBook::new(UpdatePolicy::Monotonic, 1))
            .add_order_book(symbol, OrderBook::new(UpdatePolicy::Contiguous, 2))
            .build();

        assert!(matches!(result, Err(FimlError::DuplicateOrderBook { .. })));
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
                order_book_subscribers: SubscriberRange::default(),
            }]
            .into_boxed_slice(),
            subscribers: Box::new([]),
            any_subscribers: SubscriberRange::default(),
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
                any_subscribers: SubscriberRange { start: 0, len: 1 },
            },
            order_books: OrderBookStorage::new(Vec::new()).unwrap(),
            last_timestamp: None,
        };

        let result = vector.handle_event(Event::time(1_609_459_200_000)).unwrap();

        assert_eq!(result.features_updated, 1);
        assert_eq!(vector.feature_vector().values(), [0.0, 5.0]);
    }
}

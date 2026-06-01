use crate::Float;

/// Number of [`EventKind`] variants. Used to size the per-kind feature groups in
/// [`IndicatorFeatureVector`](crate::features::IndicatorFeatureVector).
pub const EVENT_KIND_COUNT: usize = 3;

/// Kind tag of an [`Event`], used to route an event to the features that
/// subscribe to it. Discriminants must stay `0..EVENT_KIND_COUNT` and match the
/// group order in the feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Price,
    OrderBook,
    Time,
}

/// A price tick.
pub struct PriceUpdate<F: Float> {
    pub value: F,
    pub timestamp: i64,
}

/// An order-book change.
pub struct OrderBookUpdate<F: Float> {
    pub bid: F,
    pub ask: F,
    pub timestamp: i64,
}

/// A wall-clock tick carrying no market data.
pub struct TimeUpdate {
    pub timestamp: i64,
}

/// An incoming change. Each variant carries only the payload its kind needs;
/// new input streams are added as new variants rather than by widening a shared
/// struct. A feature subscribes to exactly one [`EventKind`] and is only handed
/// events of that kind.
pub enum Event<F: Float> {
    Price(PriceUpdate<F>),
    OrderBook(OrderBookUpdate<F>),
    Time(TimeUpdate),
}

impl<F: Float> Event<F> {
    /// Routing tag for this event.
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Price(_) => EventKind::Price,
            Event::OrderBook(_) => EventKind::OrderBook,
            Event::Time(_) => EventKind::Time,
        }
    }

    pub fn price(value: F, timestamp: i64) -> Self {
        Event::Price(PriceUpdate { value, timestamp })
    }

    pub fn order_book(bid: F, ask: F, timestamp: i64) -> Self {
        Event::OrderBook(OrderBookUpdate {
            bid,
            ask,
            timestamp,
        })
    }

    pub fn time(timestamp: i64) -> Self {
        Event::Time(TimeUpdate { timestamp })
    }
}

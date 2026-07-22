use std::fmt;

use crate::Float;
use crate::Symbol;

/// Number of [`EventKind`] variants. Used to size the per-kind feature groups in
/// [`IndicatorFeatureVector`](crate::features::IndicatorFeatureVector).
pub const EVENT_KIND_COUNT: usize = 5;

/// Number of dispatch groups: one per [`EventKind`] plus the "every-event" group
/// that runs on every [`dispatch`](crate::features::IndicatorFeatures::dispatch)
/// regardless of kind.
pub const FEATURE_GROUP_COUNT: usize = EVENT_KIND_COUNT + 1;

/// Index of the "every-event" group within the dispatch group table. It sits
/// after all the per-kind groups (`0..EVENT_KIND_COUNT`).
pub const EVERY_EVENT_GROUP: usize = EVENT_KIND_COUNT;

/// Kind tag of an [`Event`], used to route an event to the features that
/// subscribe to it. Discriminants must stay `0..EVENT_KIND_COUNT` and match the
/// group order in the feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventKind {
    Price,
    Volume,
    Trade,
    OrderBook,
    Time,
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Price => "price",
            Self::Volume => "volume",
            Self::Trade => "trade",
            Self::OrderBook => "order book",
            Self::Time => "time",
        };
        f.write_str(name)
    }
}

/// Where a feature subscribes in the dispatch table: to a single [`EventKind`],
/// or to **every** event. Clock features (`day_of_week`,
/// `time_since_first_event_of_day`) subscribe to every event so they refresh
/// from each event's timestamp, guaranteeing a value on every output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureRoute {
    /// Runs only for events of this kind.
    Kind(EventKind),
    /// Runs on every dispatch, whatever the event kind.
    Every,
}

impl FeatureRoute {
    /// Index of the dispatch group this route maps to: the kind's discriminant
    /// for per-kind features, or [`EVERY_EVENT_GROUP`] for every-event features.
    pub fn group_index(self) -> usize {
        match self {
            FeatureRoute::Kind(kind) => kind as usize,
            FeatureRoute::Every => EVERY_EVENT_GROUP,
        }
    }
}

/// A price tick.
pub struct PriceUpdate<F: Float> {
    pub symbol: Symbol,
    pub value: F,
    pub timestamp: i64,
}

/// A volume tick.
pub struct VolumeUpdate<F: Float> {
    pub symbol: Symbol,
    pub value: F,
    pub timestamp: i64,
}

// Who was agressor in a trade: the buyer or the seller.
// If buyer was agressor, the trade was a buy (ask) and the price is the ask price.
// If seller was agressor, the trade was a sell (bid) and the price is the bid price.
pub enum TradeSide {
    AgressorBuy,
    AgressorSell,
}

/// A trade tick carrying price and volume.
pub struct TradeUpdate<F: Float> {
    pub symbol: Symbol,
    pub price: F,
    pub volume: F,
    pub timestamp: i64,
    pub side: Option<TradeSide>,
}

/// An order-book change.
pub struct OrderBookUpdate<F: Float> {
    pub symbol: Symbol,
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
    Volume(VolumeUpdate<F>),
    Trade(TradeUpdate<F>),
    OrderBook(OrderBookUpdate<F>),
    Time(TimeUpdate),
}

impl<F: Float> Event<F> {
    /// Routing tag for this event.
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Price(_) => EventKind::Price,
            Event::Volume(_) => EventKind::Volume,
            Event::Trade(_) => EventKind::Trade,
            Event::OrderBook(_) => EventKind::OrderBook,
            Event::Time(_) => EventKind::Time,
        }
    }

    /// Timestamp carried by this event, in epoch milliseconds. Every variant
    /// carries one, so every-event (clock) features can derive calendar/day
    /// values regardless of which stream the event came from.
    pub fn timestamp(&self) -> i64 {
        match self {
            Event::Price(p) => p.timestamp,
            Event::Volume(v) => v.timestamp,
            Event::Trade(t) => t.timestamp,
            Event::OrderBook(o) => o.timestamp,
            Event::Time(t) => t.timestamp,
        }
    }

    /// Market symbol carried by this event, or `None` for the global time stream.
    pub fn symbol(&self) -> Option<Symbol> {
        match self {
            Event::Price(p) => Some(p.symbol),
            Event::Volume(v) => Some(v.symbol),
            Event::Trade(t) => Some(t.symbol),
            Event::OrderBook(o) => Some(o.symbol),
            Event::Time(_) => None,
        }
    }

    pub fn price(symbol: Symbol, value: F, timestamp: i64) -> Self {
        Event::Price(PriceUpdate {
            symbol,
            value,
            timestamp,
        })
    }

    pub fn volume(symbol: Symbol, value: F, timestamp: i64) -> Self {
        Event::Volume(VolumeUpdate {
            symbol,
            value,
            timestamp,
        })
    }

    pub fn trade(
        symbol: Symbol,
        price: F,
        volume: F,
        timestamp: i64,
        side: Option<TradeSide>,
    ) -> Self {
        Event::Trade(TradeUpdate {
            symbol,
            price,
            volume,
            timestamp,
            side,
        })
    }

    pub fn order_book(symbol: Symbol, bid: F, ask: F, timestamp: i64) -> Self {
        Event::OrderBook(OrderBookUpdate {
            symbol,
            bid,
            ask,
            timestamp,
        })
    }

    pub fn time(timestamp: i64) -> Self {
        Event::Time(TimeUpdate { timestamp })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols;

    #[test]
    fn volume_event_has_volume_kind() {
        let aapl = symbols::intern("AAPL");
        let event = Event::volume(aapl, 42.0, 123);

        assert_eq!(event.kind(), EventKind::Volume);
    }

    #[test]
    fn trade_event_has_trade_kind_and_payload() {
        let aapl = symbols::intern("AAPL");
        let event = Event::trade(aapl, 42.0, 100.0, 123, Some(TradeSide::AgressorSell));

        assert_eq!(event.kind(), EventKind::Trade);
        if let Event::Trade(trade) = event {
            assert_eq!(trade.symbol, aapl);
            assert_eq!(trade.price, 42.0);
            assert_eq!(trade.volume, 100.0);
            assert_eq!(trade.timestamp, 123);
            assert!(matches!(trade.side, Some(TradeSide::AgressorSell)));
        } else {
            unreachable!("trade constructor should return Event::Trade");
        }
    }
}

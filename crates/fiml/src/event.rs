use std::fmt;

use crate::{
    Float, Symbol,
    order_book::{OrderBookDelta, OrderBookSnapshot},
};

/// Number of [`EventKind`] variants.
pub const EVENT_KIND_COUNT: usize = 6;

/// Kind tag of an [`Event`]. Discriminants must stay in
/// `0..EVENT_KIND_COUNT` so feature routing can use them as array indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventKind {
    Price,
    Volume,
    Trade,
    OrderBookDelta,
    OrderBookSnapshot,
    Time,
}

const _: () = assert!(EventKind::Time as usize + 1 == EVENT_KIND_COUNT);

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Price => "price",
            Self::Volume => "volume",
            Self::Trade => "trade",
            Self::OrderBookDelta => "order book delta",
            Self::OrderBookSnapshot => "order book snapshot",
            Self::Time => "time",
        };
        f.write_str(name)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A wall-clock tick carrying no market data.
pub struct TimeUpdate {
    pub timestamp: i64,
}

/// Order book delta event
pub struct OrderBookDeltaEvent {
    timestamp: i64,
    symbol: Symbol,
    delta: OrderBookDelta,
}

impl OrderBookDeltaEvent {
    /// Returns the order-book mutation carried by this event.
    pub fn delta(&self) -> &OrderBookDelta {
        &self.delta
    }
}

/// A complete order-book image associated with a symbol and timestamp.
pub struct OrderBookSnapshotEvent {
    timestamp: i64,
    symbol: Symbol,
    snapshot: OrderBookSnapshot,
}

impl OrderBookSnapshotEvent {
    /// Returns the complete order-book image carried by this event.
    pub fn snapshot(&self) -> &OrderBookSnapshot {
        &self.snapshot
    }
}

/// An incoming change. Each variant carries only the payload its kind needs.
pub enum Event<F: Float> {
    Price(PriceUpdate<F>),
    Volume(VolumeUpdate<F>),
    Trade(TradeUpdate<F>),
    OrderBookDelta(OrderBookDeltaEvent),
    OrderBookSnapshot(OrderBookSnapshotEvent),
    Time(TimeUpdate),
}

impl<F: Float> Event<F> {
    /// Routing tag for this event.
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Price(_) => EventKind::Price,
            Event::Volume(_) => EventKind::Volume,
            Event::Trade(_) => EventKind::Trade,
            Event::OrderBookDelta(_) => EventKind::OrderBookDelta,
            Event::OrderBookSnapshot(_) => EventKind::OrderBookSnapshot,
            Event::Time(_) => EventKind::Time,
        }
    }

    /// Timestamp carried by this event, in epoch milliseconds.
    pub fn timestamp(&self) -> i64 {
        match self {
            Event::Price(p) => p.timestamp,
            Event::Volume(v) => v.timestamp,
            Event::Trade(t) => t.timestamp,
            Event::OrderBookDelta(o) => o.timestamp,
            Event::OrderBookSnapshot(s) => s.timestamp,
            Event::Time(t) => t.timestamp,
        }
    }

    /// Market symbol carried by this event. Time events use [`Symbol::GLOBAL`].
    pub fn symbol(&self) -> Symbol {
        match self {
            Event::Price(p) => p.symbol,
            Event::Volume(v) => v.symbol,
            Event::Trade(t) => t.symbol,
            Event::OrderBookDelta(o) => o.symbol,
            Event::OrderBookSnapshot(s) => s.symbol,
            Event::Time(_) => Symbol::GLOBAL,
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

    pub fn order_book_delta(symbol: Symbol, timestamp: i64, delta: OrderBookDelta) -> Self {
        Event::OrderBookDelta(OrderBookDeltaEvent {
            symbol,
            timestamp,
            delta,
        })
    }

    pub fn order_book_snapshot(
        symbol: Symbol,
        timestamp: i64,
        snapshot: OrderBookSnapshot,
    ) -> Self {
        Event::OrderBookSnapshot(OrderBookSnapshotEvent {
            symbol,
            timestamp,
            snapshot,
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

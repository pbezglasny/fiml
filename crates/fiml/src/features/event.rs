use crate::Float;
use crate::Symbol;

/// Number of [`EventKind`] variants. Used to size the per-kind feature groups in
/// [`IndicatorFeatureVector`](crate::features::IndicatorFeatureVector).
pub const EVENT_KIND_COUNT: usize = 5;

/// Kind tag of an [`Event`], used to route an event to the features that
/// subscribe to it. Discriminants must stay `0..EVENT_KIND_COUNT` and match the
/// group order in the feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Price,
    Volume,
    Trade,
    OrderBook,
    Time,
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

/// A trade tick carrying price and volume.
pub struct TradeUpdate<F: Float> {
    pub symbol: Symbol,
    pub price: F,
    pub volume: F,
    pub timestamp: i64,
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

    pub fn trade(symbol: Symbol, price: F, volume: F, timestamp: i64) -> Self {
        Event::Trade(TradeUpdate {
            symbol,
            price,
            volume,
            timestamp,
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

pub(crate) fn market_value_for_kind<F: Float>(
    event: &Event<F>,
    event_kind: EventKind,
    symbol: Symbol,
) -> Option<F> {
    match (event_kind, event) {
        (EventKind::Price, Event::Price(p)) if p.symbol == symbol => Some(p.value),
        (EventKind::Volume, Event::Volume(v)) if v.symbol == symbol => Some(v.value),
        (EventKind::Trade, Event::Trade(t)) if t.symbol == symbol => Some(t.price),
        _ => None,
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
        let event = Event::trade(aapl, 42.0, 100.0, 123);

        assert_eq!(event.kind(), EventKind::Trade);
        if let Event::Trade(trade) = event {
            assert_eq!(trade.symbol, aapl);
            assert_eq!(trade.price, 42.0);
            assert_eq!(trade.volume, 100.0);
            assert_eq!(trade.timestamp, 123);
        } else {
            unreachable!("trade constructor should return Event::Trade");
        }
    }
}

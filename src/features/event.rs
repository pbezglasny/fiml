use crate::Float;
use crate::Ticker;

/// Number of [`EventKind`] variants. Used to size the per-kind feature groups in
/// [`IndicatorFeatureVector`](crate::features::IndicatorFeatureVector).
pub const EVENT_KIND_COUNT: usize = 4;

/// Kind tag of an [`Event`], used to route an event to the features that
/// subscribe to it. Discriminants must stay `0..EVENT_KIND_COUNT` and match the
/// group order in the feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Price,
    Volume,
    OrderBook,
    Time,
}

/// A price tick.
pub struct PriceUpdate<F: Float> {
    pub ticker: Ticker,
    pub value: F,
    pub timestamp: i64,
}

/// A volume tick.
pub struct VolumeUpdate<F: Float> {
    pub ticker: Ticker,
    pub value: F,
    pub timestamp: i64,
}

/// An order-book change.
pub struct OrderBookUpdate<F: Float> {
    pub ticker: Ticker,
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
    OrderBook(OrderBookUpdate<F>),
    Time(TimeUpdate),
}

impl<F: Float> Event<F> {
    /// Routing tag for this event.
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Price(_) => EventKind::Price,
            Event::Volume(_) => EventKind::Volume,
            Event::OrderBook(_) => EventKind::OrderBook,
            Event::Time(_) => EventKind::Time,
        }
    }

    pub fn price(ticker: Ticker, value: F, timestamp: i64) -> Self {
        Event::Price(PriceUpdate {
            ticker,
            value,
            timestamp,
        })
    }

    pub fn volume(ticker: Ticker, value: F, timestamp: i64) -> Self {
        Event::Volume(VolumeUpdate {
            ticker,
            value,
            timestamp,
        })
    }

    pub fn order_book(ticker: Ticker, bid: F, ask: F, timestamp: i64) -> Self {
        Event::OrderBook(OrderBookUpdate {
            ticker,
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
    ticker: Ticker,
) -> Option<F> {
    match (event_kind, event) {
        (EventKind::Price, Event::Price(p)) if p.ticker == ticker => Some(p.value),
        (EventKind::Volume, Event::Volume(v)) if v.ticker == ticker => Some(v.value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ticker;

    #[test]
    fn volume_event_has_volume_kind() {
        let aapl = ticker::intern("AAPL");
        let event = Event::volume(aapl, 42.0, 123);

        assert_eq!(event.kind(), EventKind::Volume);
    }
}

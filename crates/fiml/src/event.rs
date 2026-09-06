use std::fmt;

use crate::{
    EventField, FimlError, InvalidArgumentError, Result, Symbol,
    order_book::{OrderBookDelta, OrderBookSnapshot, OrderBookUpdate, OrderBookUpdateRef},
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
pub struct PriceUpdate {
    pub symbol: Symbol,
    pub value: f64,
    pub timestamp: i64,
}

/// A volume tick.
pub struct VolumeUpdate {
    pub symbol: Symbol,
    pub value: f64,
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
pub struct TradeUpdate {
    pub symbol: Symbol,
    pub price: f64,
    pub volume: f64,
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
    pub(crate) delta: OrderBookDelta,
}

impl OrderBookDeltaEvent {
    /// Returns the order-book mutation carried by this event.
    pub fn delta(&self) -> &OrderBookDelta {
        &self.delta
    }

    pub(crate) fn into_delta(self) -> OrderBookDelta {
        self.delta
    }
}

/// A complete order-book image associated with a symbol and timestamp.
pub struct OrderBookSnapshotEvent {
    timestamp: i64,
    symbol: Symbol,
    pub(crate) snapshot: OrderBookSnapshot,
}

impl OrderBookSnapshotEvent {
    /// Returns the complete order-book image carried by this event.
    pub fn snapshot(&self) -> &OrderBookSnapshot {
        &self.snapshot
    }

    pub(crate) fn into_snapshot(self) -> OrderBookSnapshot {
        self.snapshot
    }
}

/// An incoming change. Each variant carries only the payload its kind needs.
pub enum Event {
    Price(PriceUpdate),
    Volume(VolumeUpdate),
    Trade(TradeUpdate),
    OrderBookDelta(OrderBookDeltaEvent),
    OrderBookSnapshot(OrderBookSnapshotEvent),
    Time(TimeUpdate),
}

impl Event {
    /// Rejects NaN and infinity in numeric payloads without allocating.
    ///
    /// Finite zero and negative values are accepted. Trade price is checked before
    /// volume. Time and Decimal order-book payloads need no finiteness checks.
    pub fn validate_finite_values(&self) -> Result<()> {
        let validate = |value: f64, field| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(FimlError::InvalidArgument(
                    InvalidArgumentError::NonFiniteEventValue { field },
                ))
            }
        };
        match self {
            Self::Price(update) => validate(update.value, EventField::Price),
            Self::Volume(update) => validate(update.value, EventField::Volume),
            Self::Trade(update) => {
                validate(update.price, EventField::TradePrice)?;
                validate(update.volume, EventField::TradeVolume)
            }
            Self::Time(_) | Self::OrderBookDelta(_) | Self::OrderBookSnapshot(_) => Ok(()),
        }
    }

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

    pub(crate) fn order_book_update(&self) -> Option<OrderBookUpdateRef<'_>> {
        match self {
            Self::OrderBookDelta(event) => Some(OrderBookUpdateRef::Delta(event.delta())),
            Self::OrderBookSnapshot(event) => Some(OrderBookUpdateRef::Snapshot(event.snapshot())),
            _ => None,
        }
    }

    pub(crate) fn into_order_book_update(self) -> Option<OrderBookUpdate> {
        match self {
            Self::OrderBookDelta(event) => Some(OrderBookUpdate::Delta(event.into_delta())),
            Self::OrderBookSnapshot(event) => {
                Some(OrderBookUpdate::Snapshot(event.into_snapshot()))
            }
            _ => None,
        }
    }

    pub fn price(symbol: Symbol, value: f64, timestamp: i64) -> Self {
        Event::Price(PriceUpdate {
            symbol,
            value,
            timestamp,
        })
    }

    pub fn volume(symbol: Symbol, value: f64, timestamp: i64) -> Self {
        Event::Volume(VolumeUpdate {
            symbol,
            value,
            timestamp,
        })
    }

    pub fn trade(
        symbol: Symbol,
        price: f64,
        volume: f64,
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
    fn numeric_payloads_must_be_finite() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0, 1.0] {
            for (event, field, name) in [
                (
                    Event::price(Symbol::GLOBAL, value, 0),
                    EventField::Price,
                    "price",
                ),
                (
                    Event::volume(Symbol::GLOBAL, value, 0),
                    EventField::Volume,
                    "volume",
                ),
                (
                    Event::trade(Symbol::GLOBAL, value, 1.0, 0, None),
                    EventField::TradePrice,
                    "trade price",
                ),
                (
                    Event::trade(Symbol::GLOBAL, 1.0, value, 0, None),
                    EventField::TradeVolume,
                    "trade volume",
                ),
            ] {
                let result = event.validate_finite_values();
                if value.is_finite() {
                    result.unwrap();
                } else {
                    let error = result.unwrap_err();
                    assert!(matches!(error, FimlError::InvalidArgument(
                        InvalidArgumentError::NonFiniteEventValue { field: actual }
                    ) if actual == field));
                    assert!(
                        error
                            .to_string()
                            .contains(&format!("{name} must be finite"))
                    );
                }
            }
        }
        assert!(matches!(
            Event::trade(Symbol::GLOBAL, f64::NAN, f64::INFINITY, 0, None).validate_finite_values(),
            Err(FimlError::InvalidArgument(
                InvalidArgumentError::NonFiniteEventValue {
                    field: EventField::TradePrice
                }
            ))
        ));
        Event::time(0).validate_finite_values().unwrap();
    }

    #[test]
    fn volume_event_has_volume_kind() {
        let aapl = symbols::intern("AAPL").unwrap();
        let event = Event::volume(aapl, 42.0, 123);

        assert_eq!(event.kind(), EventKind::Volume);
    }

    #[test]
    fn trade_event_has_trade_kind_and_payload() {
        let aapl = symbols::intern("AAPL").unwrap();
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

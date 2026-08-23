mod book;
mod book_side;

pub use book::*;
use rust_decimal::Decimal;

pub type OrderBookUpdateId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Describe aggregated order book level with price and size.
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

impl OrderBookLevel {
    pub fn new(price: Decimal, size: Decimal) -> Self {
        Self { price, size }
    }
}

#[derive(Clone)]
pub struct OrderBookLevelUpdate {
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
}

impl OrderBookLevelUpdate {
    pub fn new(side: Side, price: Decimal, size: Decimal) -> Self {
        Self { side, price, size }
    }
}

#[derive(Clone)]
pub struct OrderBookDelta {
    pub update_id: OrderBookUpdateId,
    pub changes: Vec<OrderBookLevelUpdate>,
}

impl OrderBookDelta {
    pub fn new(update_id: OrderBookUpdateId, changes: Vec<OrderBookLevelUpdate>) -> Self {
        Self { update_id, changes }
    }
}

/// Whole order book update.
/// Order book supposed to set all vales from event
/// and apply deltas from buffer that id greater that
/// this snapshot
pub struct OrderBookSnapshot {
    pub last_update_id: OrderBookUpdateId,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBookSnapshot {
    pub fn new(
        last_update_id: OrderBookUpdateId,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
    ) -> Self {
        Self {
            last_update_id,
            bids,
            asks,
        }
    }
}

pub enum OrderBookUpdate {
    Delta(OrderBookDelta),
    Snapshot(OrderBookSnapshot),
}

impl OrderBookUpdate {
    pub fn new_delta(update_id: OrderBookUpdateId, changes: Vec<OrderBookLevelUpdate>) -> Self {
        OrderBookUpdate::Delta(OrderBookDelta::new(update_id, changes))
    }

    pub fn new_snapshot(
        last_update_id: OrderBookUpdateId,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
    ) -> Self {
        OrderBookUpdate::Snapshot(OrderBookSnapshot {
            last_update_id,
            bids,
            asks,
        })
    }
}

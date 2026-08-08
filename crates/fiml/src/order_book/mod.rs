use std::ops::Bound::{Excluded, Included};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, VecDeque},
};

use rust_decimal::Decimal;
use serde::de;

pub type OrderBookUpdateId = u64;

/// Describe aggragated order book level with price and size.
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

pub struct DepthUntilSizeResult {
    pub price_from: Decimal,
    pub price_to: Decimal,
    pub total_size: Decimal,
}

impl OrderBookLevel {
    fn new(price: Decimal, size: Decimal) -> Self {
        Self { price, size }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

#[derive(Clone)]
pub struct OrderBookLevelUpdate {
    timestamp: i64,
    update_id: OrderBookUpdateId,
    side: Side,
    price: Decimal,
    size: Decimal,
}

pub struct OrderBookSnapshot {
    timestamp: i64,
    last_update_id: OrderBookUpdateId,
    bids: Vec<OrderBookLevel>,
    asks: Vec<OrderBookLevel>,
}

pub enum OrderBookUpdate {
    Delta(OrderBookLevelUpdate),
    Snapshot(OrderBookSnapshot),
}

#[derive(Eq, PartialEq)]
struct BookSideKey {
    side: Side,
    price: Decimal,
}

impl PartialOrd for BookSideKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.side {
            Side::Bid => Some(self.price.cmp(&other.price).reverse()),
            Side::Ask => Some(self.price.cmp(&other.price)),
        }
    }
}

impl Ord for BookSideKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.side {
            Side::Bid => self.price.cmp(&other.price).reverse(),
            Side::Ask => self.price.cmp(&other.price),
        }
    }
}

impl BookSideKey {
    fn new(side: Side, price: Decimal) -> Self {
        Self { side, price }
    }
}
struct BookSide {
    side: Side,
    levels: BTreeMap<BookSideKey, Decimal>,
}

impl BookSide {
    fn new(side: Side) -> Self {
        Self {
            side,
            levels: BTreeMap::new(),
        }
    }

    fn update_level(&mut self, price: Decimal, new_size: Decimal) {
        let key = BookSideKey::new(self.side, price);
        if new_size == Decimal::ZERO {
            self.levels.remove(&key);
        } else {
            self.levels.insert(key, new_size);
        }
    }

    fn apply_snapshot(&mut self, snapshot: Vec<OrderBookLevel>) {
        self.levels.clear();
        for level in snapshot {
            self.update_level(level.price, level.size);
        }
    }

    fn best_level(&self) -> Option<OrderBookLevel> {
        self.levels
            .first_key_value()
            .map(|(key, size)| OrderBookLevel {
                price: key.price,
                size: *size,
            })
    }

    fn get_level_size(&self, price: Decimal) -> Option<Decimal> {
        self.levels
            .get(&BookSideKey::new(self.side, price))
            .cloned()
    }

    fn top_n(&self, n: usize) -> Vec<OrderBookLevel> {
        self.levels
            .iter()
            .take(n)
            .map(|(key, size)| OrderBookLevel::new(key.price, *size))
            .collect()
    }

    fn depth_until_price(&self, price: Decimal) -> Decimal {
        let price_key = BookSideKey::new(self.side, price);
        self.levels
            .iter()
            .take_while(|(key, size)| *key <= &price_key)
            .map(|(_, size)| size)
            .sum()
    }

    fn depth_until_total_size(&self, size: Decimal) -> Option<DepthUntilSizeResult> {
        if self.levels.is_empty() {
            return None;
        }
        let mut total_size = Decimal::ZERO;
        let mut price_from = Decimal::ZERO;
        let mut price_to = Decimal::ZERO;

        for (i, level) in self.levels.iter().enumerate() {
            if i == 0 {
                price_from = level.0.price;
            }
            price_to = level.0.price;
            total_size += *level.1;

            if total_size >= size {
                break;
            }
        }
        return Some(DepthUntilSizeResult {
            price_from,
            price_to,
            total_size,
        });
    }

    fn volume_between_prices(&self, from_price: Decimal, to_price: Decimal) -> Decimal {
        let from_key = BookSideKey::new(self.side, from_price);
        let to_key = BookSideKey::new(self.side, to_price);
        self.levels
            .range((Included(from_key), Excluded(to_key)))
            .map(|(_, size)| size)
            .sum()
    }

    fn top_n_size(&self, n: usize) -> Decimal {
        self.levels.values().take(n).sum()
    }
}

pub enum UpdatePolicy {
    /// Update IDs must increase, but gaps are allowed and do not desynchronize the book.
    Monotonic,
    /// Every update ID must equal `last_update_id + 1`; a gap desynchronizes the book.
    Continious,
}

/// Result of applying of udpate to order book
pub enum UpdateOutcome {
    /// The update immediately changed the visible order book.
    Applied,
    /// The delta was retained for later replay but did not change the visible order book.
    Buffered,
    /// The update was stale or already applied, so it was ignored.
    IgnoredStale,
    /// A snapshot restored synchronization and applicable buffered deltas were replayed.
    Resynchronized,
}

pub enum OrderBookUpdateError {
    SequenceGap {
        expected: OrderBookUpdateId,
        received: OrderBookUpdateId,
    },
    BuffurCapacityExceeded {
        capacity: usize,
    },
}

pub enum SyncState {
    /// No snapshot has been applied yet; incoming deltas are buffered.
    AwaitingSnapshot,
    /// The visible order book is synchronized through `last_update_id`.
    Live { last_update_id: OrderBookUpdateId },
    /// Update continuity was lost; a fresh snapshot is required to restore synchronization.
    RequareResync { last_update_id: OrderBookUpdateId },
}

/// Order Book implemtation.
/// It supposed to store monotonic updates, updates that come out of order will be rejected
pub struct OrderBook {
    bids: BookSide,
    asks: BookSide,
    sync_state: SyncState,
    policy: UpdatePolicy,
    update_buffer: VecDeque<OrderBookLevelUpdate>,
    buffer_size: usize,
    last_snapshot_timestamp: Option<i64>,
    last_udpate_id: OrderBookUpdateId,
}

impl OrderBook {
    /// Create new order book instance
    /// Arguments:
    ///  * udpate_policy - how order book will act when received out of order updates
    ///  * buffer_size - size of history buffer to store delta updates, order book updates
    ///  return error if buffer will be full
    pub fn new(update_policy: UpdatePolicy, buffer_size: usize) -> Self {
        Self {
            bids: BookSide::new(Side::Bid),
            asks: BookSide::new(Side::Ask),
            sync_state: SyncState::AwaitingSnapshot,
            policy: update_policy,
            update_buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            last_snapshot_timestamp: None,
            last_udpate_id: 0,
        }
    }

    fn apply_delta_update(&mut self, update: &OrderBookLevelUpdate) {
        match update.side {
            Side::Bid => self.bids.update_level(update.price, update.size),
            Side::Ask => self.asks.update_level(update.price, update.size),
        }
    }

    fn apply_update_queue(&mut self) {
        self.update_buffer
            .retain(|update| update.update_id > self.last_udpate_id);
        for delta_update in &self.update_buffer {
            match delta_update.side {
                Side::Bid => self
                    .bids
                    .update_level(delta_update.price, delta_update.size),
                Side::Ask => self
                    .asks
                    .update_level(delta_update.price, delta_update.size),
            }
            self.last_udpate_id = delta_update.update_id;
        }
    }

    fn apply_snapshot(&mut self, snaphot: OrderBookSnapshot) {
        self.bids.apply_snapshot(snaphot.bids);
        self.asks.apply_snapshot(snaphot.asks);
        self.last_snapshot_timestamp = Some(snaphot.timestamp);
        self.last_udpate_id = snaphot.last_update_id;
        self.apply_update_queue();
    }

    pub fn apply_update(
        &mut self,
        update: OrderBookUpdate,
    ) -> Result<UpdateOutcome, OrderBookUpdateError> {
        match update {
            OrderBookUpdate::Delta(delta_update) => match self.sync_state {
                SyncState::AwaitingSnapshot | SyncState::RequareResync { last_update_id: _ } => {
                    self.update_buffer.push_back(delta_update);
                    return Ok(UpdateOutcome::Buffered);
                }
                SyncState::Live { last_update_id: _ } => {
                    if delta_update.update_id < self.last_udpate_id {
                        match self.policy {
                            UpdatePolicy::Monotonic => {
                                return Ok(UpdateOutcome::IgnoredStale);
                            }
                            UpdatePolicy::Continious => {
                                self.sync_state = SyncState::RequareResync {
                                    last_update_id: self.last_udpate_id,
                                };
                                return Err(OrderBookUpdateError::SequenceGap {
                                    expected: self.last_udpate_id + 1,
                                    received: delta_update.update_id,
                                });
                            }
                        }
                    }
                    if self.update_buffer.len() == self.buffer_size {
                        return Err(OrderBookUpdateError::BuffurCapacityExceeded {
                            capacity: self.buffer_size,
                        });
                    }
                    self.apply_delta_update(&delta_update);
                    self.update_buffer.push_back(delta_update);
                    return Ok(UpdateOutcome::Applied);
                }
            },
            OrderBookUpdate::Snapshot(snapshot) => {
                self.apply_snapshot(snapshot);
                Ok(UpdateOutcome::Applied)
            }
        }
    }

    pub fn last_udpate_id(&self) -> OrderBookUpdateId {
        self.last_udpate_id
    }

    pub fn last_snapshot_timestamp(&self) -> Option<i64> {
        self.last_snapshot_timestamp
    }

    fn book_side(&self, side: Side) -> &BookSide {
        match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        }
    }

    /// Returns the highest bid price currently available.
    pub fn best_bid(&self) -> Option<OrderBookLevel> {
        self.bids.best_level()
    }

    /// Returns the lowest ask price currently available.
    pub fn best_ask(&self) -> Option<OrderBookLevel> {
        self.asks.best_level()
    }

    /// Returns the midpoint between the best bid and best ask.
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Returns the absolute difference between the best ask and best bid.
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Returns the bid-ask spread expressed in basis points (1 bp = 0.01%).
    pub fn spread_bps(&self) -> Option<Decimal> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid_price)) => Some((spread / mid_price) * Decimal::from(10000)),
            _ => None,
        }
    }

    /// Get size of provided price
    pub fn level(&self, side: Side, price: Decimal) -> Option<Decimal> {
        self.book_side(side).get_level_size(price)
    }

    /// Returns the top N price levels for the specified side.
    pub fn top_n(&self, side: Side, n: usize) -> Vec<OrderBookLevel> {
        self.book_side(side).top_n(n)
    }

    /// Returns the cumulative quantity available from the best price up to the specified price.
    pub fn depth_until_price(&self, side: Side, price: Decimal) -> Decimal {
        self.book_side(side).depth_until_price(price)
    }

    /// Returns the price range and cumulative depth required to fill the specified quantity.
    pub fn depth_until_total_size(
        &self,
        side: Side,
        size: Decimal,
    ) -> Option<DepthUntilSizeResult> {
        self.book_side(side).depth_until_total_size(size)
    }

    /// Returns the total available quantity between two price levels.
    /// * `from_price` - start for interval, icldued
    /// * `to_price` - end of interval, exluded
    pub fn volume_between_prices(
        &self,
        side: Side,
        from_price: Decimal,
        to_price: Decimal,
    ) -> Decimal {
        self.book_side(side)
            .volume_between_prices(from_price, to_price)
    }

    /// Weighted mid price with values
    pub fn weighted_mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some((bid.price * bid.size + ask.price * ask.size) / (bid.size + ask.size))
            }
            _ => None,
        }
    }

    /// Microprice of best prices
    pub fn microprice(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some((bid.price * ask.size + ask.price * bid.size) / (bid.size + ask.size))
            }
            _ => None,
        }
    }

    /// Order book imbalance for n levels
    /// Values from -1 to 1, where 1 bid imbalance
    pub fn imbalance(&self, n_levels: usize) -> Option<Decimal> {
        let bid_size = self.book_side(Side::Bid).top_n_size(n_levels);
        let ask_size = self.book_side(Side::Ask).top_n_size(n_levels);

        let sum = bid_size + ask_size;
        if sum == Decimal::ZERO {
            return None;
        }
        Some((bid_size - ask_size) / (sum))
    }
}

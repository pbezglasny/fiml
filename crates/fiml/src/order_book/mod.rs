use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound::{Excluded, Included};

use rust_decimal::Decimal;

pub type OrderBookUpdateId = u64;

/// Describe aggragated order book level with price and size.
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

impl OrderBookLevel {
    pub fn new(price: Decimal, size: Decimal) -> Self {
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

pub struct OrderBookSnapshot {
    pub last_update_id: OrderBookUpdateId,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

pub enum OrderBookUpdate {
    Delta(OrderBookDelta),
    Snapshot(OrderBookSnapshot),
}

pub struct DepthUntilSizeResult {
    pub price_from: Decimal,
    pub price_to: Decimal,
    pub total_size: Decimal,
}

#[derive(Eq, PartialEq)]
struct BookSideKey {
    side: Side,
    price: Decimal,
}

impl PartialOrd for BookSideKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
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

    fn top_n(&self, n: usize) -> impl Iterator<Item = OrderBookLevel> {
        self.levels
            .iter()
            .take(n)
            .map(|(key, size)| OrderBookLevel::new(key.price, *size))
    }

    fn depth_until_price(&self, price: Decimal) -> Decimal {
        let price_key = BookSideKey::new(self.side, price);
        self.levels
            .iter()
            .take_while(|(key, ..)| *key <= &price_key)
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
        Some(DepthUntilSizeResult {
            price_from,
            price_to,
            total_size,
        })
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

#[derive(Debug, Clone, Copy)]
pub enum UpdatePolicy {
    /// Update IDs must increase, but gaps are allowed and do not desynchronize the book.
    Monotonic,
    /// Every update ID must equal `last_update_id + 1`; a gap desynchronizes the book.
    Contiguous,
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
    SnapshotHistoryGap {
        snapshot_update_id: OrderBookUpdateId,
        expected_next_id: OrderBookUpdateId,
        received: OrderBookUpdateId,
    },
    BuffurCapacityExceeded {
        capacity: usize,
    },
    StaleSnapshot,
}

pub enum SyncState {
    /// No snapshot has been applied yet; incoming deltas are buffered.
    AwaitingSnapshot,
    /// The visible order book is synchronized.
    Live,
    /// Update continuity was lost; a fresh snapshot is required to restore synchronization.
    RequareResync,
}

fn apply_delta_update(bids: &mut BookSide, asks: &mut BookSide, delta: &OrderBookDelta) {
    for level_update in &delta.changes {
        match level_update.side {
            Side::Bid => bids.update_level(level_update.price, level_update.size),
            Side::Ask => asks.update_level(level_update.price, level_update.size),
        }
    }
}

/// Order Book implemtation.
/// It supposed to store monotonic updates, updates that come out of order will be rejected
pub struct OrderBook {
    bids: BookSide,
    asks: BookSide,
    sync_state: SyncState,
    policy: UpdatePolicy,
    update_buffer: VecDeque<OrderBookDelta>,
    buffer_size: usize,
    last_update_id: OrderBookUpdateId,
    last_snapshot_update_id: OrderBookUpdateId,
}

fn has_contiguous_gap(
    policy: UpdatePolicy,
    previous_update_id: OrderBookUpdateId,
    received_update_id: OrderBookUpdateId,
) -> bool {
    matches!(policy, UpdatePolicy::Contiguous) && received_update_id != previous_update_id + 1
}

impl OrderBook {
    /// Create new order book instance
    /// Arguments:
    ///  * udpate_policy - how order book will act when received out of order updates
    ///  * buffer_size - size of history buffer to store delta updates, order book updates
    ///    return error if buffer will be full
    pub fn new(update_policy: UpdatePolicy, buffer_size: usize) -> Self {
        Self {
            bids: BookSide::new(Side::Bid),
            asks: BookSide::new(Side::Ask),
            sync_state: SyncState::AwaitingSnapshot,
            policy: update_policy,
            update_buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            last_update_id: 0,
            last_snapshot_update_id: 0,
        }
    }

    /// Check if update buffer if full.
    /// Return Ok if it has empty capacity
    /// Otherwise set sync_state to RequareResync and return Err
    fn ensure_history_buffer_capacity_or_change_sync_state(
        &mut self,
    ) -> Result<(), OrderBookUpdateError> {
        if self.update_buffer.len() == self.buffer_size {
            self.sync_state = SyncState::RequareResync;
            return Err(OrderBookUpdateError::BuffurCapacityExceeded {
                capacity: self.buffer_size,
            });
        }
        Ok(())
    }

    fn handle_contiguous_gap(
        &mut self,
        previous_update_id: OrderBookUpdateId,
        delta: OrderBookDelta,
    ) -> Result<UpdateOutcome, OrderBookUpdateError> {
        if matches!(self.sync_state, SyncState::Live) {
            self.sync_state = SyncState::RequareResync;
        }

        let delta_update_id = delta.update_id;
        self.update_buffer.push_back(delta);
        Err(OrderBookUpdateError::SequenceGap {
            expected: previous_update_id + 1,
            received: delta_update_id,
        })
    }

    fn validate_history_after_snapshot_contiguous(
        &self,
        new_snapshot_update_id: OrderBookUpdateId,
    ) -> Result<(), OrderBookUpdateError> {
        if matches!(self.policy, UpdatePolicy::Monotonic) {
            return Ok(());
        }
        let mut prev_id = new_snapshot_update_id;
        for delta in &self.update_buffer {
            if delta.update_id <= new_snapshot_update_id {
                continue;
            }
            if has_contiguous_gap(self.policy, prev_id, delta.update_id) {
                return Err(OrderBookUpdateError::SnapshotHistoryGap {
                    snapshot_update_id: new_snapshot_update_id,
                    expected_next_id: prev_id + 1,
                    received: delta.update_id,
                });
            }
            prev_id = delta.update_id;
        }
        Ok(())
    }

    fn replay_history_after_snapshot(&mut self) {
        self.update_buffer
            .retain(|update| update.update_id > self.last_snapshot_update_id);

        let Self {
            bids,
            asks,
            update_buffer,
            last_update_id,
            ..
        } = self;
        for delta in update_buffer {
            apply_delta_update(bids, asks, delta);
            *last_update_id = delta.update_id;
        }
    }

    fn apply_snapshot(
        &mut self,
        snapshot: OrderBookSnapshot,
    ) -> Result<UpdateOutcome, OrderBookUpdateError> {
        self.validate_history_after_snapshot_contiguous(snapshot.last_update_id)?;
        let was_resync = matches!(self.sync_state, SyncState::RequareResync);
        self.bids.apply_snapshot(snapshot.bids);
        self.asks.apply_snapshot(snapshot.asks);
        self.last_update_id = snapshot.last_update_id;
        self.last_snapshot_update_id = snapshot.last_update_id;
        self.replay_history_after_snapshot();
        self.sync_state = SyncState::Live;
        if was_resync {
            Ok(UpdateOutcome::Resynchronized)
        } else {
            Ok(UpdateOutcome::Applied)
        }
    }

    pub fn apply_update(
        &mut self,
        update: OrderBookUpdate,
    ) -> Result<UpdateOutcome, OrderBookUpdateError> {
        match update {
            OrderBookUpdate::Delta(delta) => {
                if let Some(previous_update_id) =
                    self.update_buffer.back().map(|delta| delta.update_id)
                {
                    if delta.update_id <= previous_update_id {
                        return Ok(UpdateOutcome::IgnoredStale);
                    }
                    if has_contiguous_gap(self.policy, previous_update_id, delta.update_id) {
                        self.ensure_history_buffer_capacity_or_change_sync_state()?;
                        return self.handle_contiguous_gap(previous_update_id, delta);
                    }
                }
                match self.sync_state {
                    SyncState::AwaitingSnapshot | SyncState::RequareResync => {
                        self.ensure_history_buffer_capacity_or_change_sync_state()?;
                        self.update_buffer.push_back(delta);
                        Ok(UpdateOutcome::Buffered)
                    }
                    SyncState::Live => {
                        if delta.update_id <= self.last_update_id {
                            return Ok(UpdateOutcome::IgnoredStale);
                        }
                        self.ensure_history_buffer_capacity_or_change_sync_state()?;
                        if has_contiguous_gap(self.policy, self.last_update_id, delta.update_id) {
                            return self.handle_contiguous_gap(self.last_update_id, delta);
                        }
                        let Self { bids, asks, .. } = self;
                        apply_delta_update(bids, asks, &delta);
                        self.last_update_id = delta.update_id;
                        self.update_buffer.push_back(delta);
                        Ok(UpdateOutcome::Applied)
                    }
                }
            }
            OrderBookUpdate::Snapshot(snapshot) => {
                if snapshot.last_update_id <= self.last_snapshot_update_id {
                    return Err(OrderBookUpdateError::StaleSnapshot);
                }
                self.apply_snapshot(snapshot)
            }
        }
    }

    pub fn last_udpate_id(&self) -> OrderBookUpdateId {
        self.last_update_id
    }

    pub fn last_snapshot_update_id(&self) -> OrderBookUpdateId {
        self.last_snapshot_update_id
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
    pub fn top_n(&self, side: Side, n: usize) -> impl Iterator<Item = OrderBookLevel> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::dec;

    fn apply_successfully(book: &mut OrderBook, update: OrderBookUpdate) -> UpdateOutcome {
        match book.apply_update(update) {
            Ok(outcome) => outcome,
            Err(_) => panic!("order-book update unexpectedly failed"),
        }
    }

    #[test]
    fn one_delta_applies_multiple_bid_and_ask_changes() {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 4);

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 101,
                changes: vec![
                    OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(5)),
                    OrderBookLevelUpdate::new(Side::Bid, dec!(99), dec!(2)),
                    OrderBookLevelUpdate::new(Side::Ask, dec!(101), dec!(4)),
                    OrderBookLevelUpdate::new(Side::Ask, dec!(102), dec!(3)),
                ],
            }),
        );
        assert!(matches!(outcome, UpdateOutcome::Buffered));

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: Vec::new(),
                asks: Vec::new(),
            }),
        );
        assert!(matches!(outcome, UpdateOutcome::Applied));

        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(5)));
        assert_eq!(book.level(Side::Bid, dec!(99)), Some(dec!(2)));
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(4)));
        assert_eq!(book.level(Side::Ask, dec!(102)), Some(dec!(3)));
        assert_eq!(book.last_udpate_id(), 101);
    }

    #[test]
    fn one_delta_updates_and_deletes_levels_together() {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 4);

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 201,
                changes: vec![
                    OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(7)),
                    OrderBookLevelUpdate::new(Side::Ask, dec!(101), dec!(0)),
                ],
            }),
        );
        assert!(matches!(outcome, UpdateOutcome::Buffered));

        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 200,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: vec![OrderBookLevel::new(dec!(101), dec!(2))],
            }),
        );

        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(7)));
        assert_eq!(book.level(Side::Ask, dec!(101)), None);
        assert_eq!(book.last_udpate_id(), 201);
    }

    #[test]
    fn snapshot_history_gap_is_rejected_without_mutation() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: vec![OrderBookLevel::new(dec!(101), dec!(2))],
            }),
        );
        assert!(matches!(outcome, UpdateOutcome::Applied));

        let gap_result = book.apply_update(OrderBookUpdate::Delta(OrderBookDelta {
            update_id: 103,
            changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(3))],
        }));
        assert!(matches!(
            gap_result,
            Err(OrderBookUpdateError::SequenceGap {
                expected: 101,
                received: 103,
            })
        ));

        let snapshot_result = book.apply_update(OrderBookUpdate::Snapshot(OrderBookSnapshot {
            last_update_id: 101,
            bids: vec![OrderBookLevel::new(dec!(90), dec!(9))],
            asks: Vec::new(),
        }));
        assert!(matches!(
            snapshot_result,
            Err(OrderBookUpdateError::SnapshotHistoryGap {
                snapshot_update_id: 101,
                expected_next_id: 102,
                received: 103,
            })
        ));

        assert!(matches!(book.sync_state, SyncState::RequareResync));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(1)));
        assert_eq!(book.level(Side::Bid, dec!(90)), None);
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(2)));
        assert_eq!(book.last_udpate_id(), 100);
        assert_eq!(book.last_snapshot_update_id(), 100);
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(103)
        );
    }

    #[test]
    fn newer_snapshot_covers_gap_and_replays_remaining_delta() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: Vec::new(),
            }),
        );
        apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 101,
                changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(2))],
            }),
        );

        let gap_result = book.apply_update(OrderBookUpdate::Delta(OrderBookDelta {
            update_id: 103,
            changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(4))],
        }));
        assert!(matches!(
            gap_result,
            Err(OrderBookUpdateError::SequenceGap {
                expected: 102,
                received: 103,
            })
        ));

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 102,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(3))],
                asks: Vec::new(),
            }),
        );

        assert!(matches!(outcome, UpdateOutcome::Resynchronized));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(4)));
        assert_eq!(book.last_udpate_id(), 103);
        assert_eq!(book.last_snapshot_update_id(), 102);
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(103)
        );
    }

    #[test]
    fn monotonic_snapshot_replays_non_contiguous_history() {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 4);

        apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 101,
                changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(1))],
            }),
        );
        apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 103,
                changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(3))],
            }),
        );

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: Vec::new(),
                asks: Vec::new(),
            }),
        );

        assert!(matches!(outcome, UpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(3)));
        assert_eq!(book.last_udpate_id(), 103);
    }

    #[test]
    fn newer_snapshot_applied_while_live_remains_live() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: Vec::new(),
            }),
        );
        apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 101,
                changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(2))],
            }),
        );

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 101,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(5))],
                asks: Vec::new(),
            }),
        );

        assert!(matches!(outcome, UpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(5)));
        assert_eq!(book.last_udpate_id(), 101);
        assert_eq!(book.last_snapshot_update_id(), 101);
        assert!(book.update_buffer.is_empty());
    }
}

use std::collections::VecDeque;
use std::fmt;

use rust_decimal::Decimal;

use crate::{
    FimlError,
    order_book::{
        OrderBookDelta, OrderBookLevel, OrderBookSnapshot, OrderBookUpdate, OrderBookUpdateId,
        OrderBookUpdateRef, Side,
    },
};

use super::book_side::BookSide;
pub struct DepthUntilSizeResult {
    pub price_from: Decimal,
    pub price_to: Decimal,
    pub total_size: Decimal,
}

#[derive(Debug, Clone, Copy)]
pub enum UpdatePolicy {
    /// Update IDs must increase, but gaps are allowed and do not desynchronize the book.
    Monotonic,
    /// Every update ID must equal `last_update_id + 1`; a gap desynchronizes the book.
    Contiguous,
}

/// Result of applying an update to the order book.
pub enum OrderBookUpdateOutcome {
    /// The update immediately changed the visible order book.
    Applied,
    /// The delta was retained for later replay but did not change the visible order book.
    Buffered,
    /// The update was stale or already applied, so it was ignored.
    IgnoredStale,
    /// A snapshot restored synchronization and applicable buffered deltas were replayed.
    Resynchronized,
}

#[derive(Debug)]
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
    BufferCapacityExceeded {
        capacity: usize,
    },
    StaleSnapshot {
        current_snapshot_update_id: OrderBookUpdateId,
        received_snapshot_update_id: OrderBookUpdateId,
    },
    InvalidUpdate {
        side: Side,
        price: Decimal,
        size: Decimal,
    },
}

impl fmt::Display for OrderBookUpdateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceGap { expected, received } => {
                write!(f, "expected update ID {expected}, received {received}")
            }
            Self::SnapshotHistoryGap {
                snapshot_update_id,
                expected_next_id,
                received,
            } => write!(
                f,
                "snapshot {snapshot_update_id} requires update ID {expected_next_id}, received {received}"
            ),
            Self::BufferCapacityExceeded { capacity } => {
                write!(
                    f,
                    "order-book update buffer capacity {capacity} was exceeded"
                )
            }
            Self::StaleSnapshot {
                current_snapshot_update_id,
                received_snapshot_update_id,
            } => write!(
                f,
                "snapshot {received_snapshot_update_id} is not newer than current snapshot {current_snapshot_update_id}"
            ),
            Self::InvalidUpdate { side, price, size } => write!(
                f,
                "invalid {side:?} level update with price {price} and size {size}"
            ),
        }
    }
}

impl std::error::Error for OrderBookUpdateError {}

impl From<OrderBookUpdateError> for FimlError {
    fn from(value: OrderBookUpdateError) -> Self {
        FimlError::OrderBookUpdateError { reason: value }
    }
}

pub enum SyncState {
    /// No snapshot has been applied yet; incoming deltas are buffered.
    AwaitingSnapshot,
    /// The visible order book is synchronized.
    Live,
    /// Update continuity was lost; a fresh snapshot is required to restore synchronization.
    RequireResync,
}

fn apply_delta_update(bids: &mut BookSide, asks: &mut BookSide, delta: &OrderBookDelta) {
    for level_update in &delta.changes {
        match level_update.side {
            Side::Bid => bids.update_level(level_update.price, level_update.size),
            Side::Ask => asks.update_level(level_update.price, level_update.size),
        }
    }
}

fn has_contiguous_gap(
    policy: UpdatePolicy,
    previous_update_id: Option<OrderBookUpdateId>,
    received_update_id: OrderBookUpdateId,
) -> bool {
    if let Some(prev_id) = previous_update_id {
        matches!(policy, UpdatePolicy::Contiguous) && received_update_id != prev_id + 1
    } else {
        false
    }
}

fn validate_level_update_deltas(update_delta: &OrderBookDelta) -> Result<(), OrderBookUpdateError> {
    for delta in &update_delta.changes {
        if delta.price < Decimal::ZERO || delta.size < Decimal::ZERO {
            return Err(OrderBookUpdateError::InvalidUpdate {
                side: delta.side,
                price: delta.price,
                size: delta.size,
            });
        }
    }
    Ok(())
}

fn validate_snapshot_update(snapshot: &OrderBookSnapshot) -> Result<(), OrderBookUpdateError> {
    fn validate_side(
        side_levels: &[OrderBookLevel],
        side: Side,
    ) -> Result<(), OrderBookUpdateError> {
        for level in side_levels {
            if level.price < Decimal::ZERO || level.size < Decimal::ZERO {
                return Err(OrderBookUpdateError::InvalidUpdate {
                    side,
                    price: level.price,
                    size: level.size,
                });
            }
        }
        Ok(())
    }
    validate_side(&snapshot.bids, Side::Bid)?;
    validate_side(&snapshot.asks, Side::Ask)?;
    Ok(())
}

/// Order book implementation.
/// It supposed to store monotonic updates, updates that come out of order will be rejected
///
/// Current implementation does not handle overflow of maximal values of UpdateId and could causes errors.
/// User supposed to check provided update_id passed into `apply_update` method
pub struct OrderBook {
    bids: BookSide,
    asks: BookSide,
    sync_state: SyncState,
    policy: UpdatePolicy,
    update_buffer: VecDeque<OrderBookDelta>,
    buffer_size: usize,
    last_update_id: Option<OrderBookUpdateId>,
    last_snapshot_update_id: Option<OrderBookUpdateId>,
}

/// Allocation-free plan for applying an update after all fallible checks finish.
pub(crate) struct PreparedOrderBookUpdate {
    action: PreparedUpdateAction,
}

enum PreparedUpdateAction {
    IgnoreStaleDelta,
    BufferDelta,
    ApplyDelta,
    ApplySnapshot {
        resynchronized: bool,
    },
    Reject {
        error: OrderBookUpdateError,
        action: RejectedUpdateAction,
    },
}

enum RejectedUpdateAction {
    None,
    RequireResync,
    BufferDelta { require_resync: bool },
}

impl PreparedOrderBookUpdate {
    fn new(action: PreparedUpdateAction) -> Self {
        Self { action }
    }

    fn reject(error: OrderBookUpdateError, action: RejectedUpdateAction) -> Self {
        Self::new(PreparedUpdateAction::Reject { error, action })
    }

    pub(crate) fn is_rejected(&self) -> bool {
        matches!(self.action, PreparedUpdateAction::Reject { .. })
    }
}

impl OrderBook {
    /// Create new order book instance
    /// Arguments:
    ///  * update_policy - how order book will act when receiving out-of-order updates
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
            last_update_id: None,
            last_snapshot_update_id: None,
        }
    }

    fn reject_full_history_buffer(&self) -> PreparedOrderBookUpdate {
        PreparedOrderBookUpdate::reject(
            OrderBookUpdateError::BufferCapacityExceeded {
                capacity: self.buffer_size,
            },
            RejectedUpdateAction::RequireResync,
        )
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
            if has_contiguous_gap(self.policy, Some(prev_id), delta.update_id) {
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

    fn replay_history_after_snapshot(&mut self, last_snapshot_id: OrderBookUpdateId) {
        self.update_buffer
            .retain(|update| update.update_id > last_snapshot_id);

        let Self {
            bids,
            asks,
            update_buffer,
            last_update_id,
            ..
        } = self;
        for delta in update_buffer {
            apply_delta_update(bids, asks, delta);
            *last_update_id = Some(delta.update_id);
        }
    }

    fn apply_snapshot(&mut self, snapshot: OrderBookSnapshot, resynchronized: bool) {
        debug_assert_eq!(
            resynchronized,
            matches!(self.sync_state, SyncState::RequireResync),
            "prepared snapshot outcome must be committed without intervening book updates"
        );
        self.bids.apply_snapshot(snapshot.bids);
        self.asks.apply_snapshot(snapshot.asks);
        self.last_update_id = Some(snapshot.last_update_id);
        self.last_snapshot_update_id = Some(snapshot.last_update_id);
        self.replay_history_after_snapshot(snapshot.last_update_id);
        self.sync_state = SyncState::Live;
    }

    /// Determines an update's exact mutation without changing the book.
    pub(crate) fn prepare_update(&self, update: OrderBookUpdateRef<'_>) -> PreparedOrderBookUpdate {
        match update {
            OrderBookUpdateRef::Delta(delta) => self.prepare_delta_update(delta),
            OrderBookUpdateRef::Snapshot(snapshot) => self.prepare_snapshot_update(snapshot),
        }
    }

    fn prepare_delta_update(&self, delta: &OrderBookDelta) -> PreparedOrderBookUpdate {
        if let Err(error) = validate_level_update_deltas(delta) {
            return PreparedOrderBookUpdate::reject(error, RejectedUpdateAction::None);
        }

        if let Some(previous_update_id) = self.update_buffer.back().map(|delta| delta.update_id) {
            if delta.update_id <= previous_update_id {
                return PreparedOrderBookUpdate::new(PreparedUpdateAction::IgnoreStaleDelta);
            }
            if has_contiguous_gap(self.policy, Some(previous_update_id), delta.update_id) {
                if self.update_buffer.len() == self.buffer_size {
                    return self.reject_full_history_buffer();
                }
                return PreparedOrderBookUpdate::reject(
                    OrderBookUpdateError::SequenceGap {
                        expected: previous_update_id + 1,
                        received: delta.update_id,
                    },
                    RejectedUpdateAction::BufferDelta {
                        require_resync: matches!(self.sync_state, SyncState::Live),
                    },
                );
            }
        }

        match self.sync_state {
            SyncState::AwaitingSnapshot | SyncState::RequireResync => {
                if self.update_buffer.len() == self.buffer_size {
                    self.reject_full_history_buffer()
                } else {
                    PreparedOrderBookUpdate::new(PreparedUpdateAction::BufferDelta)
                }
            }
            SyncState::Live => {
                if delta.update_id <= self.last_update_id.unwrap_or(0) {
                    return PreparedOrderBookUpdate::new(PreparedUpdateAction::IgnoreStaleDelta);
                }
                if self.update_buffer.len() == self.buffer_size {
                    return self.reject_full_history_buffer();
                }
                if has_contiguous_gap(self.policy, self.last_update_id, delta.update_id) {
                    return PreparedOrderBookUpdate::reject(
                        OrderBookUpdateError::SequenceGap {
                            expected: self.last_update_id.unwrap_or(0) + 1,
                            received: delta.update_id,
                        },
                        RejectedUpdateAction::BufferDelta {
                            require_resync: true,
                        },
                    );
                }
                PreparedOrderBookUpdate::new(PreparedUpdateAction::ApplyDelta)
            }
        }
    }

    fn prepare_snapshot_update(&self, snapshot: &OrderBookSnapshot) -> PreparedOrderBookUpdate {
        if let Err(error) = validate_snapshot_update(snapshot) {
            return PreparedOrderBookUpdate::reject(error, RejectedUpdateAction::None);
        }
        if let Some(previous_snapshot_id) = self.last_snapshot_update_id
            && snapshot.last_update_id <= previous_snapshot_id
        {
            return PreparedOrderBookUpdate::reject(
                OrderBookUpdateError::StaleSnapshot {
                    current_snapshot_update_id: previous_snapshot_id,
                    received_snapshot_update_id: snapshot.last_update_id,
                },
                RejectedUpdateAction::None,
            );
        }
        if let Err(error) = self.validate_history_after_snapshot_contiguous(snapshot.last_update_id)
        {
            return PreparedOrderBookUpdate::reject(error, RejectedUpdateAction::None);
        }
        PreparedOrderBookUpdate::new(PreparedUpdateAction::ApplySnapshot {
            resynchronized: matches!(self.sync_state, SyncState::RequireResync),
        })
    }

    /// Applies a previously prepared update without repeating fallible checks.
    pub(crate) fn commit_update(
        &mut self,
        prepared: PreparedOrderBookUpdate,
        update: OrderBookUpdate,
    ) -> Result<OrderBookUpdateOutcome, OrderBookUpdateError> {
        match (prepared.action, update) {
            (PreparedUpdateAction::IgnoreStaleDelta, OrderBookUpdate::Delta(_)) => {
                Ok(OrderBookUpdateOutcome::IgnoredStale)
            }
            (PreparedUpdateAction::BufferDelta, OrderBookUpdate::Delta(delta)) => {
                self.update_buffer.push_back(delta);
                Ok(OrderBookUpdateOutcome::Buffered)
            }
            (PreparedUpdateAction::ApplyDelta, OrderBookUpdate::Delta(delta)) => {
                apply_delta_update(&mut self.bids, &mut self.asks, &delta);
                self.last_update_id = Some(delta.update_id);
                self.update_buffer.push_back(delta);
                Ok(OrderBookUpdateOutcome::Applied)
            }
            (
                PreparedUpdateAction::ApplySnapshot { resynchronized },
                OrderBookUpdate::Snapshot(snapshot),
            ) => {
                self.apply_snapshot(snapshot, resynchronized);
                if resynchronized {
                    Ok(OrderBookUpdateOutcome::Resynchronized)
                } else {
                    Ok(OrderBookUpdateOutcome::Applied)
                }
            }
            (
                PreparedUpdateAction::Reject {
                    error,
                    action: RejectedUpdateAction::None,
                },
                _,
            ) => Err(error),
            (
                PreparedUpdateAction::Reject {
                    error,
                    action: RejectedUpdateAction::RequireResync,
                },
                _,
            ) => {
                self.sync_state = SyncState::RequireResync;
                Err(error)
            }
            (
                PreparedUpdateAction::Reject {
                    error,
                    action: RejectedUpdateAction::BufferDelta { require_resync },
                },
                OrderBookUpdate::Delta(delta),
            ) => {
                if require_resync {
                    self.sync_state = SyncState::RequireResync;
                }
                self.update_buffer.push_back(delta);
                Err(error)
            }
            _ => unreachable!("prepared update must be committed with the update it describes"),
        }
    }

    /// Update order book by passing delta or entire snapshot of order book.
    pub fn apply_update(
        &mut self,
        update: OrderBookUpdate,
    ) -> Result<OrderBookUpdateOutcome, OrderBookUpdateError> {
        let prepared = self.prepare_update(update.as_ref());
        self.commit_update(prepared, update)
    }

    pub fn last_update_id(&self) -> Option<OrderBookUpdateId> {
        self.last_update_id
    }

    pub fn last_snapshot_update_id(&self) -> Option<OrderBookUpdateId> {
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
            (Some(bid), Some(ask)) => Some((ask.price - bid.price).abs()),
            _ => None,
        }
    }

    /// Returns the bid-ask spread expressed in basis points (1 bp = 0.01%).
    pub fn spread_bps(&self) -> Option<Decimal> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid_price)) => {
                if mid_price == Decimal::ZERO {
                    None
                } else {
                    Some((spread / mid_price) * Decimal::from(10000))
                }
            }
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
    /// * `from_price` - start of interval, included
    /// * `to_price` - end of interval, excluded
    pub fn volume_between_prices(
        &self,
        side: Side,
        from_price: Decimal,
        to_price: Decimal,
    ) -> Result<Decimal, FimlError> {
        if from_price >= to_price {
            return Err(FimlError::InvalidPriceRange {
                from_price,
                to_price,
            });
        }
        Ok(self
            .book_side(side)
            .volume_between_prices(from_price, to_price))
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
    use crate::order_book::OrderBookLevelUpdate;

    use super::*;
    use rust_decimal::dec;

    fn apply_successfully(book: &mut OrderBook, update: OrderBookUpdate) -> OrderBookUpdateOutcome {
        match book.apply_update(update) {
            Ok(outcome) => outcome,
            Err(_) => panic!("order-book update unexpectedly failed"),
        }
    }

    fn bid_delta(update_id: OrderBookUpdateId, size: Decimal) -> OrderBookUpdate {
        OrderBookUpdate::Delta(OrderBookDelta {
            update_id,
            changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), size)],
        })
    }

    fn bid_snapshot(last_update_id: OrderBookUpdateId, size: Decimal) -> OrderBookUpdate {
        OrderBookUpdate::Snapshot(OrderBookSnapshot {
            last_update_id,
            bids: vec![OrderBookLevel::new(dec!(100), size)],
            asks: Vec::new(),
        })
    }

    fn indicator_book() -> OrderBook {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 8);
        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: vec![
                    OrderBookLevel::new(dec!(99), dec!(2)),
                    OrderBookLevel::new(dec!(98), dec!(3)),
                    OrderBookLevel::new(dec!(97), dec!(5)),
                ],
                asks: vec![
                    OrderBookLevel::new(dec!(101), dec!(6)),
                    OrderBookLevel::new(dec!(102), dec!(4)),
                    OrderBookLevel::new(dec!(103), dec!(10)),
                ],
            }),
        );
        book
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
        assert!(matches!(outcome, OrderBookUpdateOutcome::Buffered));

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: Vec::new(),
                asks: Vec::new(),
            }),
        );
        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));

        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(5)));
        assert_eq!(book.level(Side::Bid, dec!(99)), Some(dec!(2)));
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(4)));
        assert_eq!(book.level(Side::Ask, dec!(102)), Some(dec!(3)));
        assert_eq!(book.last_update_id(), Some(101));
    }

    #[test]
    fn snapshot_replays_multiple_buffered_contiguous_deltas_in_order() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 101,
                changes: vec![
                    OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(1)),
                    OrderBookLevelUpdate::new(Side::Ask, dec!(101), dec!(3)),
                ],
            }),
        );
        assert!(matches!(outcome, OrderBookUpdateOutcome::Buffered));

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Delta(OrderBookDelta {
                update_id: 102,
                changes: vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(2))],
            }),
        );
        assert!(matches!(outcome, OrderBookUpdateOutcome::Buffered));
        assert_eq!(book.level(Side::Bid, dec!(100)), None);
        assert_eq!(book.level(Side::Ask, dec!(101)), None);

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: Vec::new(),
                asks: Vec::new(),
            }),
        );

        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(2)));
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(3)));
        assert_eq!(book.last_update_id(), Some(102));
        assert_eq!(book.last_snapshot_update_id(), Some(100));
        assert_eq!(book.update_buffer.len(), 2);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(101)
        );
        assert_eq!(
            book.update_buffer.back().map(|delta| delta.update_id),
            Some(102)
        );
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
        assert!(matches!(outcome, OrderBookUpdateOutcome::Buffered));

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
        assert_eq!(book.last_update_id(), Some(201));
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
        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));

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

        assert!(matches!(book.sync_state, SyncState::RequireResync));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(1)));
        assert_eq!(book.level(Side::Bid, dec!(90)), None);
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(2)));
        assert_eq!(book.last_update_id(), Some(100));
        assert_eq!(book.last_snapshot_update_id(), Some(100));
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

        assert!(matches!(outcome, OrderBookUpdateOutcome::Resynchronized));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(4)));
        assert_eq!(book.last_update_id(), Some(103));
        assert_eq!(book.last_snapshot_update_id(), Some(102));
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

        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(3)));
        assert_eq!(book.last_update_id(), Some(103));
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

        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(5)));
        assert_eq!(book.last_update_id(), Some(101));
        assert_eq!(book.last_snapshot_update_id(), Some(101));
        assert!(book.update_buffer.is_empty());
    }

    #[test]
    fn first_snapshot_with_zero_id_is_accepted() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        assert_eq!(book.last_update_id(), None);
        assert_eq!(book.last_snapshot_update_id(), None);

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 0,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: Vec::new(),
            }),
        );

        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.last_update_id(), Some(0));
        assert_eq!(book.last_snapshot_update_id(), Some(0));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(1)));
    }

    #[test]
    fn duplicate_delta_is_ignored_without_mutation() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        apply_successfully(&mut book, bid_snapshot(100, dec!(1)));
        apply_successfully(&mut book, bid_delta(101, dec!(2)));

        let outcome = apply_successfully(&mut book, bid_delta(101, dec!(9)));

        assert!(matches!(outcome, OrderBookUpdateOutcome::IgnoredStale));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(2)));
        assert_eq!(book.last_update_id(), Some(101));
        assert_eq!(book.last_snapshot_update_id(), Some(100));
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer
                .front()
                .and_then(|delta| delta.changes.first())
                .map(|change| change.size),
            Some(dec!(2))
        );
    }

    #[test]
    fn stale_delta_is_ignored_without_mutation() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        apply_successfully(&mut book, bid_snapshot(100, dec!(1)));
        apply_successfully(&mut book, bid_delta(101, dec!(2)));

        let outcome = apply_successfully(&mut book, bid_delta(99, dec!(9)));

        assert!(matches!(outcome, OrderBookUpdateOutcome::IgnoredStale));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(2)));
        assert_eq!(book.last_update_id(), Some(101));
        assert_eq!(book.last_snapshot_update_id(), Some(100));
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(101)
        );
    }

    #[test]
    fn stale_snapshots_are_rejected_without_mutation() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 4);

        apply_successfully(&mut book, bid_snapshot(100, dec!(1)));
        apply_successfully(&mut book, bid_delta(101, dec!(2)));

        let equal_snapshot_result = book.apply_update(bid_snapshot(100, dec!(9)));
        assert!(matches!(
            equal_snapshot_result,
            Err(OrderBookUpdateError::StaleSnapshot {
                current_snapshot_update_id: 100,
                received_snapshot_update_id: 100,
            })
        ));

        let older_snapshot_result = book.apply_update(bid_snapshot(99, dec!(8)));
        assert!(matches!(
            older_snapshot_result,
            Err(OrderBookUpdateError::StaleSnapshot {
                current_snapshot_update_id: 100,
                received_snapshot_update_id: 99,
            })
        ));

        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(2)));
        assert_eq!(book.last_update_id(), Some(101));
        assert_eq!(book.last_snapshot_update_id(), Some(100));
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(101)
        );
    }

    #[test]
    fn snapshot_older_than_visible_book_replays_uncovered_history() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 8);

        apply_successfully(&mut book, bid_snapshot(100, dec!(1)));
        apply_successfully(&mut book, bid_delta(101, dec!(2)));
        apply_successfully(&mut book, bid_delta(102, dec!(3)));
        apply_successfully(&mut book, bid_delta(103, dec!(4)));

        let outcome = apply_successfully(&mut book, bid_snapshot(102, dec!(30)));

        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert!(matches!(book.sync_state, SyncState::Live));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(4)));
        assert_eq!(book.last_update_id(), Some(103));
        assert_eq!(book.last_snapshot_update_id(), Some(102));
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(103)
        );
    }

    #[test]
    fn history_capacity_is_enforced_before_first_snapshot() {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 1);

        let outcome = apply_successfully(&mut book, bid_delta(1, dec!(1)));
        assert!(matches!(outcome, OrderBookUpdateOutcome::Buffered));

        let result = book.apply_update(bid_delta(2, dec!(2)));
        assert!(matches!(
            result,
            Err(OrderBookUpdateError::BufferCapacityExceeded { capacity: 1 })
        ));

        assert!(matches!(book.sync_state, SyncState::RequireResync));
        assert_eq!(book.level(Side::Bid, dec!(100)), None);
        assert_eq!(book.last_update_id(), None);
        assert_eq!(book.last_snapshot_update_id(), None);
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(1)
        );
    }

    #[test]
    fn history_capacity_is_enforced_before_live_book_mutation() {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 1);

        apply_successfully(&mut book, bid_snapshot(100, dec!(1)));
        apply_successfully(&mut book, bid_delta(101, dec!(2)));

        let result = book.apply_update(bid_delta(102, dec!(3)));
        assert!(matches!(
            result,
            Err(OrderBookUpdateError::BufferCapacityExceeded { capacity: 1 })
        ));

        assert!(matches!(book.sync_state, SyncState::RequireResync));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(2)));
        assert_eq!(book.last_update_id(), Some(101));
        assert_eq!(book.last_snapshot_update_id(), Some(100));
        assert_eq!(book.update_buffer.len(), 1);
        assert_eq!(
            book.update_buffer.front().map(|delta| delta.update_id),
            Some(101)
        );
    }

    #[test]
    fn negative_delta_values_are_rejected_atomically() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 8);

        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: vec![OrderBookLevel::new(dec!(101), dec!(2))],
            }),
        );

        let negative_size_result = book.apply_update(OrderBookUpdate::Delta(OrderBookDelta {
            update_id: 101,
            changes: vec![
                OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(9)),
                OrderBookLevelUpdate::new(Side::Ask, dec!(101), dec!(-1)),
            ],
        }));
        assert!(matches!(
            negative_size_result,
            Err(OrderBookUpdateError::InvalidUpdate {
                side: Side::Ask,
                price,
                size,
            }) if price == dec!(101) && size == dec!(-1)
        ));

        let negative_price_result = book.apply_update(OrderBookUpdate::Delta(OrderBookDelta {
            update_id: 101,
            changes: vec![
                OrderBookLevelUpdate::new(Side::Ask, dec!(101), dec!(9)),
                OrderBookLevelUpdate::new(Side::Bid, dec!(-1), dec!(1)),
            ],
        }));
        assert!(matches!(
            negative_price_result,
            Err(OrderBookUpdateError::InvalidUpdate {
                side: Side::Bid,
                price,
                size,
            }) if price == dec!(-1) && size == dec!(1)
        ));

        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(1)));
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(2)));
        assert_eq!(book.last_update_id(), Some(100));
        assert_eq!(book.last_snapshot_update_id(), Some(100));

        let outcome = apply_successfully(&mut book, bid_delta(101, dec!(3)));
        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(3)));
        assert_eq!(book.last_update_id(), Some(101));
    }

    #[test]
    fn negative_snapshot_values_are_rejected_atomically() {
        let mut book = OrderBook::new(UpdatePolicy::Contiguous, 8);

        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 100,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(1))],
                asks: vec![OrderBookLevel::new(dec!(101), dec!(2))],
            }),
        );
        apply_successfully(&mut book, bid_delta(101, dec!(2)));
        apply_successfully(&mut book, bid_delta(102, dec!(3)));

        let negative_size_result =
            book.apply_update(OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 101,
                bids: vec![OrderBookLevel::new(dec!(90), dec!(9))],
                asks: vec![OrderBookLevel::new(dec!(101), dec!(-1))],
            }));
        assert!(matches!(
            negative_size_result,
            Err(OrderBookUpdateError::InvalidUpdate {
                side: Side::Ask,
                price,
                size,
            }) if price == dec!(101) && size == dec!(-1)
        ));

        let negative_price_result =
            book.apply_update(OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 101,
                bids: vec![
                    OrderBookLevel::new(dec!(90), dec!(9)),
                    OrderBookLevel::new(dec!(-1), dec!(1)),
                ],
                asks: Vec::new(),
            }));
        assert!(matches!(
            negative_price_result,
            Err(OrderBookUpdateError::InvalidUpdate {
                side: Side::Bid,
                price,
                size,
            }) if price == dec!(-1) && size == dec!(1)
        ));

        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(3)));
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(2)));
        assert_eq!(book.level(Side::Bid, dec!(90)), None);
        assert_eq!(book.last_update_id(), Some(102));
        assert_eq!(book.last_snapshot_update_id(), Some(100));

        let outcome = apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 101,
                bids: vec![OrderBookLevel::new(dec!(100), dec!(20))],
                asks: vec![OrderBookLevel::new(dec!(101), dec!(5))],
            }),
        );
        assert!(matches!(outcome, OrderBookUpdateOutcome::Applied));
        assert_eq!(book.level(Side::Bid, dec!(100)), Some(dec!(3)));
        assert_eq!(book.level(Side::Ask, dec!(101)), Some(dec!(5)));
        assert_eq!(book.last_update_id(), Some(102));
        assert_eq!(book.last_snapshot_update_id(), Some(101));
    }

    #[test]
    fn invalid_volume_price_range_reports_its_bounds() {
        let book = OrderBook::new(UpdatePolicy::Monotonic, 1);

        let error = book
            .volume_between_prices(Side::Ask, dec!(101), dec!(100))
            .expect_err("reversed price bounds must be rejected");

        assert!(matches!(
            &error,
            FimlError::InvalidPriceRange {
                from_price,
                to_price,
            } if *from_price == dec!(101) && *to_price == dec!(100)
        ));
        assert_eq!(
            error.to_string(),
            "invalid price range: from price 101 must be less than to price 100"
        );
    }

    #[test]
    fn volume_between_prices_uses_numeric_half_open_bounds_for_both_sides() {
        let book = indicator_book();

        let bid_volume = book
            .volume_between_prices(Side::Bid, dec!(97), dec!(99))
            .expect("valid bid range must be accepted");
        let ask_volume = book
            .volume_between_prices(Side::Ask, dec!(101), dec!(103))
            .expect("valid ask range must be accepted");

        assert_eq!(bid_volume, dec!(8));
        assert_eq!(ask_volume, dec!(10));
    }

    #[test]
    fn best_levels_and_top_n_follow_side_price_priority() {
        let book = indicator_book();

        let best_bid = book.best_bid().expect("fixture has bid levels");
        let best_ask = book.best_ask().expect("fixture has ask levels");
        let top_bids: Vec<_> = book
            .top_n(Side::Bid, 2)
            .map(|level| (level.price, level.size))
            .collect();
        let top_asks: Vec<_> = book
            .top_n(Side::Ask, 2)
            .map(|level| (level.price, level.size))
            .collect();

        assert_eq!((best_bid.price, best_bid.size), (dec!(99), dec!(2)));
        assert_eq!((best_ask.price, best_ask.size), (dec!(101), dec!(6)));
        assert_eq!(top_bids, vec![(dec!(99), dec!(2)), (dec!(98), dec!(3))]);
        assert_eq!(top_asks, vec![(dec!(101), dec!(6)), (dec!(102), dec!(4))]);
    }

    #[test]
    fn midpoint_and_spreads_use_the_best_prices() {
        let book = indicator_book();

        assert_eq!(book.mid_price(), Some(dec!(100)));
        assert_eq!(book.spread(), Some(dec!(2)));
        assert_eq!(book.spread_bps(), Some(dec!(200)));
    }

    #[test]
    fn spread_bps_is_unavailable_for_a_zero_midpoint() {
        let mut book = OrderBook::new(UpdatePolicy::Monotonic, 1);
        apply_successfully(
            &mut book,
            OrderBookUpdate::Snapshot(OrderBookSnapshot {
                last_update_id: 1,
                bids: vec![OrderBookLevel::new(dec!(0), dec!(1))],
                asks: vec![OrderBookLevel::new(dec!(0), dec!(1))],
            }),
        );

        assert_eq!(book.mid_price(), Some(dec!(0)));
        assert_eq!(book.spread_bps(), None);
    }

    #[test]
    fn depth_until_price_accumulates_from_the_best_level() {
        let book = indicator_book();

        assert_eq!(book.depth_until_price(Side::Bid, dec!(98)), dec!(5));
        assert_eq!(book.depth_until_price(Side::Ask, dec!(102)), dec!(10));
    }

    #[test]
    fn depth_until_total_size_requires_positive_fillable_size() {
        let book = indicator_book();

        let bid_depth = book
            .depth_until_total_size(Side::Bid, dec!(4))
            .expect("five units across two bid levels fill the request");
        let ask_depth = book
            .depth_until_total_size(Side::Ask, dec!(7))
            .expect("ten units across two ask levels fill the request");

        assert_eq!(bid_depth.price_from, dec!(99));
        assert_eq!(bid_depth.price_to, dec!(98));
        assert_eq!(bid_depth.total_size, dec!(5));
        assert_eq!(ask_depth.price_from, dec!(101));
        assert_eq!(ask_depth.price_to, dec!(102));
        assert_eq!(ask_depth.total_size, dec!(10));
        assert!(book.depth_until_total_size(Side::Bid, dec!(0)).is_none());
        assert!(book.depth_until_total_size(Side::Bid, dec!(-1)).is_none());
        assert!(book.depth_until_total_size(Side::Bid, dec!(11)).is_none());
    }

    #[test]
    fn weighted_mid_price_uses_best_level_sizes() {
        let book = indicator_book();

        assert_eq!(book.weighted_mid_price(), Some(dec!(100.5)));
    }

    #[test]
    fn microprice_cross_weights_best_level_prices() {
        let book = indicator_book();

        assert_eq!(book.microprice(), Some(dec!(99.5)));
    }

    #[test]
    fn imbalance_uses_the_requested_number_of_levels() {
        let book = indicator_book();

        assert_eq!(book.imbalance(1), Some(dec!(-0.5)));
        assert_eq!(book.imbalance(0), None);
    }

    #[test]
    fn price_and_size_indicators_are_unavailable_without_levels() {
        let book = OrderBook::new(UpdatePolicy::Monotonic, 1);

        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
        assert_eq!(book.mid_price(), None);
        assert_eq!(book.spread(), None);
        assert_eq!(book.spread_bps(), None);
        assert_eq!(book.weighted_mid_price(), None);
        assert_eq!(book.microprice(), None);
        assert_eq!(book.imbalance(1), None);
    }
}

//! Maintains synchronized bid/ask levels and exposes quote and depth queries.

mod book;
mod book_side;
mod dense_book_side;

pub use book_side::{BTreeBookSide, BookSideStorage};
pub use dense_book_side::{DenseBookConfig, DenseBookSide};

pub(crate) use book::PreparedOrderBookUpdate;
pub use book::{
    DepthUntilSizeResult, OrderBook, OrderBookUpdateError, OrderBookUpdateOutcome, UpdatePolicy,
};
use rust_decimal::Decimal;

/// Sequence identifier supplied by the market-data feed.
pub type OrderBookUpdateId = u64;

/// Side of a two-sided order book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// Resting buy orders.
    Bid,
    /// Resting sell orders.
    Ask,
}

/// Aggregated price and size for one level of a snapshot.
#[derive(Clone)]
pub struct OrderBookLevel {
    /// Nonnegative price of the level.
    pub price: Decimal,
    /// Nonnegative size; zero removes this price level.
    pub size: Decimal,
}

impl OrderBookLevel {
    /// Creates a level; price and size are validated when the update is applied.
    pub fn new(price: Decimal, size: Decimal) -> Self {
        Self { price, size }
    }
}

/// Absolute size replacement for one side and price; a zero size deletes the level.
#[derive(Clone)]
pub struct OrderBookLevelUpdate {
    /// Side containing the changed level.
    pub side: Side,
    /// Nonnegative price of the level.
    pub price: Decimal,
    /// Nonnegative size; zero removes this price level.
    pub size: Decimal,
}

impl OrderBookLevelUpdate {
    /// Creates an absolute size update; zero size deletes the level when applied.
    pub fn new(side: Side, price: Decimal, size: Decimal) -> Self {
        Self { side, price, size }
    }
}

/// Sequenced collection of absolute level-size changes.
#[derive(Clone)]
pub struct OrderBookDelta {
    /// Sequence ID of this delta.
    pub update_id: OrderBookUpdateId,
    /// Absolute level sizes to apply in order, not size increments.
    pub changes: Vec<OrderBookLevelUpdate>,
}

impl OrderBookDelta {
    /// Creates a delta from a sequence ID and ordered level changes.
    pub fn new(update_id: OrderBookUpdateId, changes: Vec<OrderBookLevelUpdate>) -> Self {
        Self { update_id, changes }
    }
}

/// Complete book image through a sequence ID.
/// Applying it replaces visible levels and replays eligible newer buffered deltas.
#[derive(Clone)]
pub struct OrderBookSnapshot {
    /// Latest delta sequence ID included in this snapshot.
    pub last_update_id: OrderBookUpdateId,
    /// Complete bid levels.
    pub bids: Vec<OrderBookLevel>,
    /// Complete ask levels.
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBookSnapshot {
    /// Creates a snapshot containing all levels through `last_update_id`.
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

/// Owned delta or snapshot passed to an order book.
#[derive(Clone)]
pub enum OrderBookUpdate {
    /// Incremental changes to existing levels.
    Delta(OrderBookDelta),
    /// Complete replacement image used to synchronize the book.
    Snapshot(OrderBookSnapshot),
}

/// Borrowed form of an order-book update used for allocation-free preparation.
pub(crate) enum OrderBookUpdateRef<'a> {
    Delta(&'a OrderBookDelta),
    Snapshot(&'a OrderBookSnapshot),
}

impl OrderBookUpdate {
    /// Creates a delta update without applying or validating its levels.
    pub fn new_delta(update_id: OrderBookUpdateId, changes: Vec<OrderBookLevelUpdate>) -> Self {
        OrderBookUpdate::Delta(OrderBookDelta::new(update_id, changes))
    }

    /// Creates a full snapshot update without applying or validating its levels.
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

    pub(crate) fn as_ref(&self) -> OrderBookUpdateRef<'_> {
        match self {
            Self::Delta(delta) => OrderBookUpdateRef::Delta(delta),
            Self::Snapshot(snapshot) => OrderBookUpdateRef::Snapshot(snapshot),
        }
    }
}

/// Serializable construction parameters for a fresh per-symbol order book.
/// Live levels and synchronization history are intentionally excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderBookConfig {
    /// Non-global market symbol assigned to the book.
    pub symbol: crate::Symbol,
    /// Rule used to detect missing or stale delta sequence IDs.
    pub update_policy: UpdatePolicy,
    /// Maximum number of deltas retained for synchronization and snapshot replay.
    pub buffer_size: usize,
    /// Optional preallocated tick grid; omission selects BTreeMap storage.
    pub dense: Option<DenseBookConfig>,
}

impl OrderBookConfig {
    /// Selects synchronization policy and delta-history capacity for a symbol.
    pub const fn new(
        symbol: crate::Symbol,
        update_policy: UpdatePolicy,
        buffer_size: usize,
    ) -> Self {
        Self {
            symbol,
            update_policy,
            buffer_size,
            dense: None,
        }
    }

    /// Selects dense storage; grid validity is checked when attaching this configuration to a spec.
    pub const fn with_dense(
        mut self,
        tick_size: Decimal,
        min_price: Decimal,
        max_price: Decimal,
    ) -> Self {
        self.dense = Some(DenseBookConfig {
            tick_size,
            min_price,
            max_price,
        });
        self
    }

    /// Constructs fresh storage and synchronization state using this configuration.
    pub fn build(&self) -> Result<OrderBook, OrderBookUpdateError> {
        match self.dense {
            Some(grid) => OrderBook::new_dense(
                self.update_policy,
                self.buffer_size,
                grid.tick_size,
                grid.min_price,
                grid.max_price,
            ),
            None => Ok(OrderBook::new(self.update_policy, self.buffer_size)),
        }
    }
}

//! Stores one price-ordered side of an order book and answers depth queries.
//!
use std::collections::BTreeMap;

use rust_decimal::Decimal;

use crate::order_book::{OrderBookLevel, Side};

use super::{DenseBookSide, DepthUntilSizeResult, OrderBookUpdateError};

use std::ops::Bound::{Excluded, Included};
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
/// Sparse price-level storage backed by a BTreeMap.
pub struct BTreeBookSide {
    side: Side,
    levels: BTreeMap<BookSideKey, Decimal>,
}

impl BTreeBookSide {
    /// Creates an empty side with best-price-first ordering.
    pub fn new(side: Side) -> Self {
        Self {
            side,
            levels: BTreeMap::new(),
        }
    }
}

impl BookSideStorage for BTreeBookSide {
    fn side(&self) -> Side {
        self.side
    }

    fn update_level(&mut self, price: Decimal, new_size: Decimal) {
        let key = BookSideKey::new(self.side, price);
        if new_size == Decimal::ZERO {
            self.levels.remove(&key);
        } else {
            self.levels.insert(key, new_size);
        }
    }

    fn clear(&mut self) {
        self.levels.clear();
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

    fn volume_between_prices(&self, from_price: Decimal, to_price: Decimal) -> Decimal {
        let from_key = BookSideKey::new(self.side, from_price);
        let to_key = BookSideKey::new(self.side, to_price);
        let bounds = match self.side {
            Side::Bid => (Excluded(to_key), Included(from_key)),
            Side::Ask => (Included(from_key), Excluded(to_key)),
        };
        self.levels.range(bounds).map(|(_, size)| size).sum()
    }

    fn top_n_size(&self, n: usize) -> Decimal {
        self.levels.values().take(n).sum()
    }
}

/// Storage for one side, allowing dense and sparse levels to share book synchronization.
/// Implementations iterate occupied prices from best to worst and treat zero size as deletion.
pub trait BookSideStorage {
    /// Identifies the ordering of this side.
    fn side(&self) -> Side;

    /// Checks a level before any part of its containing update is applied.
    fn validate_level(&self, price: Decimal, size: Decimal) -> Result<(), OrderBookUpdateError> {
        if price < Decimal::ZERO || size < Decimal::ZERO {
            return Err(OrderBookUpdateError::InvalidUpdate {
                side: self.side(),
                price,
                size,
            });
        }
        Ok(())
    }

    /// Replaces a level. The caller must first pass it through `validate_level`.
    fn update_level(&mut self, price: Decimal, size: Decimal);

    /// Removes all occupied levels, retaining reusable storage where possible.
    fn clear(&mut self);

    /// Looks up an occupied price; missing prices return `None`.
    fn get_level_size(&self, price: Decimal) -> Option<Decimal>;

    /// Visits at most `n` occupied levels, best first, without allocating.
    fn top_n(&self, n: usize) -> impl Iterator<Item = OrderBookLevel>;

    /// Returns the best occupied level.
    fn best_level(&self) -> Option<OrderBookLevel> {
        self.top_n(1).next()
    }

    /// Sums size from the best price through the inclusive price threshold.
    fn depth_until_price(&self, price: Decimal) -> Decimal {
        self.top_n(usize::MAX)
            .take_while(|level| match self.side() {
                Side::Bid => level.price >= price,
                Side::Ask => level.price <= price,
            })
            .map(|level| level.size)
            .sum()
    }

    /// Returns the whole levels needed to reach a positive target size.
    fn depth_until_total_size(&self, size: Decimal) -> Option<DepthUntilSizeResult> {
        if size <= Decimal::ZERO {
            return None;
        }
        let mut levels = self.top_n(usize::MAX);
        let first = levels.next()?;
        let mut result = DepthUntilSizeResult {
            price_from: first.price,
            price_to: first.price,
            total_size: first.size,
        };
        if result.total_size >= size {
            return Some(result);
        }
        for level in levels {
            result.price_to = level.price;
            result.total_size += level.size;
            if result.total_size >= size {
                return Some(result);
            }
        }
        None
    }

    /// Sums sizes in `[from_price, to_price)`; the caller must ensure `from_price < to_price`.
    fn volume_between_prices(&self, from_price: Decimal, to_price: Decimal) -> Decimal {
        self.top_n(usize::MAX)
            .filter(|level| level.price >= from_price && level.price < to_price)
            .map(|level| level.size)
            .sum()
    }

    /// Sums at most `n` occupied levels from the best price.
    fn top_n_size(&self, n: usize) -> Decimal {
        self.top_n(n).map(|level| level.size).sum()
    }
}

/// Selects storage without boxing either the side or its query iterators.
pub(crate) enum BookSide {
    Tree(BTreeBookSide),
    Dense(DenseBookSide),
}

impl BookSide {
    pub(crate) fn new(side: Side) -> Self {
        Self::Tree(BTreeBookSide::new(side))
    }

    pub(crate) fn apply_snapshot(&mut self, snapshot: Vec<OrderBookLevel>) {
        self.clear();
        for level in snapshot {
            self.update_level(level.price, level.size);
        }
    }
}

impl BookSideStorage for BookSide {
    fn side(&self) -> Side {
        match self {
            Self::Tree(side) => side.side(),
            Self::Dense(side) => side.side(),
        }
    }
    fn validate_level(&self, price: Decimal, size: Decimal) -> Result<(), OrderBookUpdateError> {
        match self {
            Self::Tree(side) => side.validate_level(price, size),
            Self::Dense(side) => side.validate_level(price, size),
        }
    }
    fn update_level(&mut self, price: Decimal, size: Decimal) {
        match self {
            Self::Tree(side) => side.update_level(price, size),
            Self::Dense(side) => side.update_level(price, size),
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Tree(side) => side.clear(),
            Self::Dense(side) => side.clear(),
        }
    }
    fn get_level_size(&self, price: Decimal) -> Option<Decimal> {
        match self {
            Self::Tree(side) => side.get_level_size(price),
            Self::Dense(side) => side.get_level_size(price),
        }
    }
    fn top_n(&self, n: usize) -> impl Iterator<Item = OrderBookLevel> {
        let (tree, dense) = match self {
            Self::Tree(side) => (Some(side.top_n(n)), None),
            Self::Dense(side) => (None, Some(side.top_n(n))),
        };
        tree.into_iter()
            .flatten()
            .chain(dense.into_iter().flatten())
    }
    fn best_level(&self) -> Option<OrderBookLevel> {
        match self {
            Self::Tree(side) => side.best_level(),
            Self::Dense(side) => side.best_level(),
        }
    }
    fn volume_between_prices(&self, from: Decimal, to: Decimal) -> Decimal {
        match self {
            Self::Tree(side) => side.volume_between_prices(from, to),
            Self::Dense(side) => side.volume_between_prices(from, to),
        }
    }
    fn top_n_size(&self, n: usize) -> Decimal {
        match self {
            Self::Tree(side) => side.top_n_size(n),
            Self::Dense(side) => side.top_n_size(n),
        }
    }
}

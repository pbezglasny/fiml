//! Stores one size per tick in a bounded, preallocated price grid.

use rust_decimal::{Decimal, prelude::ToPrimitive};

use super::{BookSideStorage, OrderBookLevel, OrderBookUpdateError, Side};

/// Dense storage for one side of an order book, with zero marking an unoccupied tick.
/// Level updates and queries do not allocate after construction. Traversal skips empty ticks.
pub struct DenseBookSide {
    side: Side,
    tick_size: Decimal,
    min_price: Decimal,
    max_price: Decimal,
    sizes: Vec<Decimal>,
    occupied: Option<(usize, usize)>,
}

impl DenseBookSide {
    /// Preallocates the inclusive price range; both bounds must be nonnegative tick multiples.
    /// Returns an error for invalid grids, unrepresentable lengths, or allocation failure.
    pub fn new(
        side: Side,
        tick_size: Decimal,
        min_price: Decimal,
        max_price: Decimal,
    ) -> Result<Self, OrderBookUpdateError> {
        let len = DenseBookConfig {
            tick_size,
            min_price,
            max_price,
        }
        .level_count()?;
        let mut sizes = Vec::new();
        sizes
            .try_reserve_exact(len)
            .map_err(|_| OrderBookUpdateError::DenseCapacityExceeded)?;
        sizes.resize(len, Decimal::ZERO);
        Ok(Self {
            side,
            tick_size,
            min_price,
            max_price,
            sizes,
            occupied: None,
        })
    }

    fn index(&self, price: Decimal) -> Option<usize> {
        if price < self.min_price
            || price > self.max_price
            || price.checked_rem(self.tick_size) != Some(Decimal::ZERO)
        {
            return None;
        }
        price
            .checked_sub(self.min_price)?
            .checked_div(self.tick_size)?
            .to_usize()
            .filter(|&index| index < self.sizes.len())
    }

    fn level_at(&self, index: usize) -> OrderBookLevel {
        OrderBookLevel::new(
            self.min_price + self.tick_size * Decimal::from(index),
            self.sizes[index],
        )
    }
}

impl BookSideStorage for DenseBookSide {
    fn side(&self) -> Side {
        self.side
    }

    fn validate_level(&self, price: Decimal, size: Decimal) -> Result<(), OrderBookUpdateError> {
        if size < Decimal::ZERO || self.index(price).is_none() {
            return Err(OrderBookUpdateError::InvalidUpdate {
                side: self.side,
                price,
                size,
            });
        }
        Ok(())
    }

    fn update_level(&mut self, price: Decimal, size: Decimal) {
        let index = self
            .index(price)
            .expect("dense level must be validated before mutation");
        self.sizes[index] = size;
        if size != Decimal::ZERO {
            self.occupied = Some(match self.occupied {
                Some((first, last)) => (first.min(index), last.max(index)),
                None => (index, index),
            });
        } else if let Some((mut first, mut last)) = self.occupied {
            // ponytail: boundary deletion scans empty ticks; add an occupancy bitmap if profiling warrants it.
            while first < last && self.sizes[first] == Decimal::ZERO {
                first += 1;
            }
            while last > first && self.sizes[last] == Decimal::ZERO {
                last -= 1;
            }
            self.occupied = (self.sizes[first] != Decimal::ZERO).then_some((first, last));
        }
    }

    fn clear(&mut self) {
        self.sizes.fill(Decimal::ZERO);
        self.occupied = None;
    }

    fn get_level_size(&self, price: Decimal) -> Option<Decimal> {
        self.index(price)
            .map(|index| self.sizes[index])
            .filter(|size| *size != Decimal::ZERO)
    }

    fn best_level(&self) -> Option<OrderBookLevel> {
        self.occupied.map(|(first, last)| {
            self.level_at(match self.side {
                Side::Bid => last,
                Side::Ask => first,
            })
        })
    }

    fn top_n(&self, n: usize) -> impl Iterator<Item = OrderBookLevel> {
        let mut indices = self.occupied.map_or(0..0, |(first, last)| first..last + 1);
        std::iter::from_fn(move || match self.side {
            Side::Bid => indices.next_back(),
            Side::Ask => indices.next(),
        })
        .filter(|&index| self.sizes[index] != Decimal::ZERO)
        .map(|index| self.level_at(index))
        .take(n)
    }
}

/// Bounds and tick size for preallocating dense price levels in a configured book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseBookConfig {
    /// Positive spacing between adjacent price slots.
    pub tick_size: Decimal,
    /// Inclusive nonnegative lower bound, aligned to the tick size.
    pub min_price: Decimal,
    /// Inclusive upper bound, aligned to the tick size.
    pub max_price: Decimal,
}

impl DenseBookConfig {
    pub(crate) fn level_count(&self) -> Result<usize, OrderBookUpdateError> {
        let Self {
            tick_size,
            min_price,
            max_price,
        } = *self;
        let invalid = || OrderBookUpdateError::InvalidDenseConfiguration {
            tick_size,
            min_price,
            max_price,
        };
        if tick_size <= Decimal::ZERO
            || min_price < Decimal::ZERO
            || max_price < min_price
            || min_price.checked_rem(tick_size) != Some(Decimal::ZERO)
            || max_price.checked_rem(tick_size) != Some(Decimal::ZERO)
        {
            return Err(invalid());
        }
        let len = max_price
            .checked_sub(min_price)
            .and_then(|span| span.checked_div(tick_size))
            .and_then(|ticks| ticks.to_usize())
            .and_then(|ticks| ticks.checked_add(1))
            .ok_or_else(invalid)?;
        // Every intermediate tick must be representable, even near Decimal's precision limit.
        if len > 1
            && max_price
                .checked_sub(tick_size)
                .and_then(|previous| max_price.checked_sub(previous))
                != Some(tick_size)
        {
            return Err(invalid());
        }
        Ok(len)
    }
}

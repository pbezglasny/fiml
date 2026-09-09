//! Converts exact Python book payloads once before handing them to the core runtime.

use fiml::order_book::{
    OrderBookDelta, OrderBookLevel, OrderBookLevelUpdate, OrderBookSnapshot, OrderBookUpdate, Side,
};
use fiml::{Event, Symbol};
use pyo3::{exceptions::PyValueError, prelude::*};
use rust_decimal::Decimal;

pub(crate) fn decimal(value: &str) -> PyResult<Decimal> {
    let value = Decimal::from_str_exact(value)
        .map_err(|error| PyValueError::new_err(format!("invalid exact decimal: {error}")))?;
    if value < Decimal::ZERO {
        return Err(PyValueError::new_err(
            "book prices and sizes must be nonnegative",
        ));
    }
    Ok(value)
}

pub(crate) fn side(value: &str) -> PyResult<Side> {
    match value {
        "bid" => Ok(Side::Bid),
        "ask" => Ok(Side::Ask),
        _ => Err(PyValueError::new_err("book side must be bid or ask")),
    }
}

/// Immutable, validated snapshot or delta with exact decimal price/size values.
/// Construction owns payload allocation; both Python runtimes reuse core dispatch.
#[pyclass(frozen)]
pub struct OrderBookEvent {
    pub(crate) symbol: Symbol,
    pub(crate) timestamp: i64,
    update: OrderBookUpdate,
}

impl OrderBookEvent {
    pub(crate) fn event(&self) -> Event {
        match &self.update {
            OrderBookUpdate::Delta(delta) => {
                Event::order_book_delta(self.symbol, self.timestamp, delta.clone())
            }
            OrderBookUpdate::Snapshot(snapshot) => {
                Event::order_book_snapshot(self.symbol, self.timestamp, snapshot.clone())
            }
        }
    }
}

#[pymethods]
impl OrderBookEvent {
    /// Builds a snapshot from (price, size) pairs of exact decimal strings.
    #[staticmethod]
    fn snapshot(
        symbol: &str,
        timestamp: i64,
        update_id: u64,
        bids: Vec<[String; 2]>,
        asks: Vec<[String; 2]>,
    ) -> PyResult<Self> {
        fn levels(values: Vec<[String; 2]>) -> PyResult<Vec<OrderBookLevel>> {
            values
                .into_iter()
                .map(|[price, size]| Ok(OrderBookLevel::new(decimal(&price)?, decimal(&size)?)))
                .collect()
        }
        Ok(Self {
            symbol: super::intern_symbol(symbol)?,
            timestamp,
            update: OrderBookUpdate::Snapshot(OrderBookSnapshot::new(
                update_id,
                levels(bids)?,
                levels(asks)?,
            )),
        })
    }

    /// Builds a delta from (side, price, size) triples; zero sizes delete levels.
    #[staticmethod]
    fn delta(
        symbol: &str,
        timestamp: i64,
        update_id: u64,
        changes: Vec<[String; 3]>,
    ) -> PyResult<Self> {
        let changes = changes
            .into_iter()
            .map(|[book_side, price, size]| {
                Ok(OrderBookLevelUpdate::new(
                    side(&book_side)?,
                    decimal(&price)?,
                    decimal(&size)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            symbol: super::intern_symbol(symbol)?,
            timestamp,
            update: OrderBookUpdate::Delta(OrderBookDelta::new(update_id, changes)),
        })
    }
}

//! Python bindings for the `fiml` indicator engine.
//!
//! The bindings deliberately run the *exact* Rust engine: features are computed
//! by replaying events through [`DynIndicatorEngine::dispatch`], the same code
//! the live Rust environment uses. Build both sides from the same
//! [`EngineSpec`] JSON and feed the same events in the same order to get
//! identical output. The engine is `f64` only.

use fiml::{DynIndicatorEngine, EngineSpec, Event, IndicatorFeatures, Symbol, symbols};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Event-kind codes for the columnar `transform`/`update` API. They mirror the
/// engine's event kinds. Each kind reads only the payload columns it needs (see
/// [`Engine::build_event`]); `OrderBook` dispatches fine even though no builtin
/// feature subscribes to it yet (the dispatch is a no-op until one does).
const KIND_PRICE: u8 = 0;
const KIND_VOLUME: u8 = 1;
const KIND_TRADE: u8 = 2;
const KIND_ORDERBOOK: u8 = 3;
const KIND_TIME: u8 = 4;

/// A configured, runnable feature engine.
#[pyclass]
pub struct Engine {
    inner: DynIndicatorEngine,
    /// Handle (index) -> interned symbol, so Python can pass cheap integer ids
    /// in array columns instead of strings per row.
    symbols: Vec<Symbol>,
    n_features: usize,
}

impl Engine {
    fn symbol_at(&self, handle: i64) -> PyResult<Symbol> {
        usize::try_from(handle)
            .ok()
            .and_then(|index| self.symbols.get(index).copied())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown symbol handle {handle}; call Engine.symbol(name) first"
                ))
            })
    }

    /// Build an [`Event`] from the row's event kind and whichever payload
    /// columns it needs. Each kind reads only its required columns; a missing
    /// required column is a `ValueError`. This is the single source of truth for
    /// the kind -> field mapping shared by [`update`](Self::update) (scalars) and
    /// [`transform`](Self::transform) (one row of its columns).
    // The argument list mirrors the engine's event payload fields and the
    // per-kind columns of the Python API, so it stays flat by design.
    #[allow(clippy::too_many_arguments)]
    fn build_event(
        &self,
        kind: u8,
        symbol: i64,
        timestamp: i64,
        price: Option<f64>,
        volume: Option<f64>,
        bid: Option<f64>,
        ask: Option<f64>,
    ) -> PyResult<Event<f64>> {
        Ok(match kind {
            KIND_PRICE => {
                Event::price(self.symbol_at(symbol)?, require("price", price)?, timestamp)
            }
            KIND_VOLUME => Event::volume(
                self.symbol_at(symbol)?,
                require("volume", volume)?,
                timestamp,
            ),
            KIND_TRADE => Event::trade(
                self.symbol_at(symbol)?,
                require("price", price)?,
                require("volume", volume)?,
                timestamp,
            ),
            KIND_ORDERBOOK => Event::order_book(
                self.symbol_at(symbol)?,
                require("bid", bid)?,
                require("ask", ask)?,
                timestamp,
            ),
            KIND_TIME => Event::time(timestamp),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported event kind {other} \
                     (expected 0=price, 1=volume, 2=trade, 3=orderbook, 4=time)"
                )));
            }
        })
    }
}

/// Fetch a payload value an event kind requires, erroring with the column name
/// when the caller did not supply that column.
fn require(column: &str, value: Option<f64>) -> PyResult<f64> {
    value.ok_or_else(|| PyValueError::new_err(format!("event kind requires the `{column}` column")))
}

/// Resolve an optional `transform` payload column to a contiguous slice, checking
/// that a supplied column matches the row count. The returned slice borrows the
/// array for as long as `array` is held, so the per-row loop only indexes it.
fn column<'a>(
    name: &str,
    array: &'a Option<PyReadonlyArray1<'_, f64>>,
    n_rows: usize,
) -> PyResult<Option<&'a [f64]>> {
    array
        .as_ref()
        .map(|array| {
            let slice = array.as_slice()?;
            if slice.len() != n_rows {
                return Err(PyValueError::new_err(format!(
                    "the `{name}` column must match the length of `kind`"
                )));
            }
            Ok(slice)
        })
        .transpose()
}

#[pymethods]
impl Engine {
    /// Build an engine from an `EngineSpec` JSON string (the parity contract
    /// shared with the live Rust environment).
    #[staticmethod]
    fn from_spec_json(json: &str) -> PyResult<Self> {
        let spec: EngineSpec =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = DynIndicatorEngine::from_spec(&spec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let n_features = inner.feature_names().len();
        Ok(Self {
            inner,
            symbols: Vec::new(),
            n_features,
        })
    }

    /// Intern `name` and return a stable integer handle to use in the `symbol`
    /// column of [`transform`](Self::transform) / [`update`](Self::update).
    fn symbol(&mut self, name: &str) -> usize {
        let symbol = symbols::intern(name);
        if let Some(index) = self.symbols.iter().position(|s| *s == symbol) {
            return index;
        }
        self.symbols.push(symbol);
        self.symbols.len() - 1
    }

    /// Feature (column) names in output order.
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names()
    }

    /// Number of feature columns.
    fn n_features(&self) -> usize {
        self.n_features
    }

    /// Current feature values in output order.
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Apply a single event and update the feature vector. Useful for live
    /// stepping and for checking parity against [`transform`](Self::transform).
    ///
    /// Pass only the payload values the event kind needs (see
    /// [`transform`](Self::transform) for the per-kind columns): e.g.
    /// `update(KIND_PRICE, sym, ts, price=...)` or
    /// `update(KIND_ORDERBOOK, sym, ts, bid=..., ask=...)`.
    #[pyo3(signature = (kind, symbol, timestamp, *, price=None, volume=None, bid=None, ask=None))]
    #[allow(clippy::too_many_arguments)] // payload columns are the Python keyword API
    fn update(
        &mut self,
        kind: u8,
        symbol: i64,
        timestamp: i64,
        price: Option<f64>,
        volume: Option<f64>,
        bid: Option<f64>,
        ask: Option<f64>,
    ) -> PyResult<()> {
        let event = self.build_event(kind, symbol, timestamp, price, volume, bid, ask)?;
        self.inner.dispatch(&event);
        Ok(())
    }

    /// Replay a full event stream and return one feature row per input row.
    ///
    /// `kind`, `symbol` and `timestamp` are required and equal length; the
    /// payload columns are optional and each row reads only the columns its kind
    /// needs:
    ///
    /// - `KIND_PRICE` -> `price`
    /// - `KIND_VOLUME` -> `volume`
    /// - `KIND_TRADE` -> `price` and `volume`
    /// - `KIND_ORDERBOOK` -> `bid` and `ask`
    /// - `KIND_TIME` -> none
    ///
    /// A row whose kind needs a column that was not supplied raises a
    /// `ValueError` naming that column. Any provided payload column must match
    /// the length of `kind`. Row `i` builds its event, dispatches it, then
    /// snapshots every feature into row `i` of the returned `(n_rows,
    /// n_features)` `float64` matrix. Looping in Rust keeps this fast while using
    /// the exact live dispatch path.
    #[pyo3(signature = (kind, symbol, timestamp, *, price=None, volume=None, bid=None, ask=None))]
    #[allow(clippy::too_many_arguments)] // payload columns are the Python keyword API
    fn transform<'py>(
        &mut self,
        py: Python<'py>,
        kind: PyReadonlyArray1<'py, u8>,
        symbol: PyReadonlyArray1<'py, i64>,
        timestamp: PyReadonlyArray1<'py, i64>,
        price: Option<PyReadonlyArray1<'py, f64>>,
        volume: Option<PyReadonlyArray1<'py, f64>>,
        bid: Option<PyReadonlyArray1<'py, f64>>,
        ask: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let kind = kind.as_slice()?;
        let symbol = symbol.as_slice()?;
        let timestamp = timestamp.as_slice()?;
        let n_rows = kind.len();

        if symbol.len() != n_rows || timestamp.len() != n_rows {
            return Err(PyValueError::new_err(
                "kind, symbol and timestamp must have the same length",
            ));
        }

        // Resolve each optional payload column to a slice once, validating its
        // length here so the per-row hot loop only indexes (no allocation).
        let price = column("price", &price, n_rows)?;
        let volume = column("volume", &volume, n_rows)?;
        let bid = column("bid", &bid, n_rows)?;
        let ask = column("ask", &ask, n_rows)?;

        let n_features = self.n_features;
        let mut out = vec![0.0f64; n_rows * n_features];
        for row in 0..n_rows {
            let event = self.build_event(
                kind[row],
                symbol[row],
                timestamp[row],
                price.map(|column| column[row]),
                volume.map(|column| column[row]),
                bid.map(|column| column[row]),
                ask.map(|column| column[row]),
            )?;
            self.inner.dispatch(&event);
            out[row * n_features..(row + 1) * n_features].copy_from_slice(self.inner.values());
        }

        let matrix = Array2::from_shape_vec((n_rows, n_features), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(matrix.into_pyarray(py))
    }
}

#[pymodule]
fn _fiml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add("KIND_PRICE", KIND_PRICE)?;
    m.add("KIND_VOLUME", KIND_VOLUME)?;
    m.add("KIND_TRADE", KIND_TRADE)?;
    m.add("KIND_ORDERBOOK", KIND_ORDERBOOK)?;
    m.add("KIND_TIME", KIND_TIME)?;
    Ok(())
}

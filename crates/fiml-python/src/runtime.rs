//! Shares event validation, replay, output buffers, and runtime plumbing.

use std::collections::HashMap;

use fiml::{
    Event, FeatureExtractor as RustFeatureExtractor, FeatureVector, FimlError,
    Pipeline as RustPipeline, Symbol, TradeSide, VecFeatureVector,
};
use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::feature_extractor_spec::FeatureExtractorSpec;
use crate::intern_symbol;
use crate::order_book::OrderBookEvent;

/// Event-kind codes for the columnar `transform`/`update` API. They mirror the
/// extractor's event kinds. Each kind reads only the payload columns it needs.
/// Book payloads use `OrderBookEvent` and the dedicated book replay methods.
pub(crate) const KIND_PRICE: u8 = 0;
pub(crate) const KIND_VOLUME: u8 = 1;
pub(crate) const KIND_TRADE: u8 = 2;
pub(crate) const KIND_ORDERBOOK: u8 = 3;
pub(crate) const KIND_TIME: u8 = 4;

/// Trade-side codes for optional `side` payloads on trade events.
pub(crate) const SIDE_AGGRESSOR_BUY: u8 = 0;
pub(crate) const SIDE_AGGRESSOR_SELL: u8 = 1;

#[derive(Clone, Copy)]
pub(crate) enum OutputDtype {
    Float32,
    Float64,
}

impl OutputDtype {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            _ => Err(Self::invalid()),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }

    fn invalid() -> PyErr {
        PyValueError::new_err(
            "output_dtype must be \"float32\", \"float64\", numpy.float32, or numpy.float64",
        )
    }
}

enum OutputBuffer {
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl OutputBuffer {
    fn new(dtype: OutputDtype, len: usize) -> Self {
        match dtype {
            OutputDtype::Float32 => Self::Float32(vec![0.0; len]),
            OutputDtype::Float64 => Self::Float64(vec![0.0; len]),
        }
    }

    fn write_row(&mut self, row: usize, row_width: usize, values: &[f64]) {
        let range = row * row_width..(row + 1) * row_width;
        match self {
            Self::Float32(output) => {
                for (target, &value) in output[range].iter_mut().zip(values) {
                    *target = value as f32;
                }
            }
            Self::Float64(output) => output[range].copy_from_slice(values),
        }
    }

    fn into_pyarray(self, py: Python<'_>, n_rows: usize, n_features: usize) -> PyResult<Py<PyAny>> {
        match self {
            Self::Float32(output) => Array2::from_shape_vec((n_rows, n_features), output)
                .map_err(|error| PyValueError::new_err(error.to_string()))
                .map(|matrix| matrix.into_pyarray(py).into_any().unbind()),
            Self::Float64(output) => Array2::from_shape_vec((n_rows, n_features), output)
                .map_err(|error| PyValueError::new_err(error.to_string()))
                .map(|matrix| matrix.into_pyarray(py).into_any().unbind()),
        }
    }
}

pub(crate) type CoreFeatureExtractor = RustFeatureExtractor<VecFeatureVector>;
pub(crate) type CorePipeline = RustPipeline<VecFeatureVector, VecFeatureVector>;

pub(crate) fn build_core(
    feature_extractor_spec: &FeatureExtractorSpec,
) -> PyResult<CoreFeatureExtractor> {
    let output_vector = VecFeatureVector::new_of_length(
        feature_extractor_spec.core.feature_vector_capacity(),
        feature_extractor_spec.core.feature_vector_length(),
    );
    feature_extractor_spec
        .core
        .build(output_vector)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn complete_names(active_ids: &[String], capacity: usize) -> Vec<String> {
    let mut names = active_ids.to_vec();
    names.extend((active_ids.len()..capacity).map(|index| format!("__reserved_{index}")));
    names
}

/// Common event-processing interface used by the Python runtime driver.
///
/// This allows raw feature extractors and model-input pipelines to share event
/// validation, replay, and output buffering while retaining their concrete
/// Rust runtime types.
pub(crate) trait EventRuntime {
    fn has_order_book(&self, symbol: Symbol) -> bool;
    fn handle_event(&mut self, event: Event) -> fiml::Result<()>;
    fn last_timestamp(&self) -> Option<i64>;
    fn last_timestamp_for_symbol(&self, symbol: Symbol) -> Option<i64>;
    fn values(&self) -> &[f64];
}

impl EventRuntime for CoreFeatureExtractor {
    fn has_order_book(&self, symbol: Symbol) -> bool {
        self.order_book_of_symbol(symbol).is_some()
    }
    fn handle_event(&mut self, event: Event) -> fiml::Result<()> {
        RustFeatureExtractor::handle_event(self, event).map(|_| ())
    }

    fn last_timestamp(&self) -> Option<i64> {
        RustFeatureExtractor::last_timestamp(self)
    }

    fn last_timestamp_for_symbol(&self, symbol: Symbol) -> Option<i64> {
        RustFeatureExtractor::last_timestamp_for_symbol(self, symbol)
    }

    fn values(&self) -> &[f64] {
        self.feature_vector().values()
    }
}

impl EventRuntime for CorePipeline {
    fn has_order_book(&self, symbol: Symbol) -> bool {
        self.order_book_of_symbol(symbol).is_some()
    }
    fn handle_event(&mut self, event: Event) -> fiml::Result<()> {
        RustPipeline::handle_event(self, event).map(|_| ())
    }

    fn last_timestamp(&self) -> Option<i64> {
        RustPipeline::last_timestamp(self)
    }

    fn last_timestamp_for_symbol(&self, symbol: Symbol) -> Option<i64> {
        RustPipeline::last_timestamp_for_symbol(self, symbol)
    }

    fn values(&self) -> &[f64] {
        RustPipeline::values(self)
    }
}

pub(crate) struct RuntimeDriver<R>
where
    R: EventRuntime,
{
    pub(crate) inner: R,
    symbols: Vec<Symbol>,
    pub(crate) feature_names: Vec<String>,
    pub(crate) raw_feature_names: Vec<String>,
    pub(crate) active_feature_count: usize,
    pub(crate) output_dtype: OutputDtype,
    runtime_name: &'static str,
    lock_subject: &'static str,
}

pub(crate) struct RuntimeLayout {
    pub(crate) active_ids: Vec<String>,
    pub(crate) capacity: usize,
    pub(crate) raw_active_ids: Vec<String>,
    pub(crate) raw_capacity: usize,
    pub(crate) runtime_name: &'static str,
    pub(crate) lock_subject: &'static str,
}

impl<R> RuntimeDriver<R>
where
    R: EventRuntime,
{
    pub(crate) fn new(inner: R, output_dtype: OutputDtype, layout: RuntimeLayout) -> Self {
        let active_feature_count = layout.active_ids.len();
        Self {
            inner,
            symbols: Vec::new(),
            feature_names: complete_names(&layout.active_ids, layout.capacity),
            raw_feature_names: complete_names(&layout.raw_active_ids, layout.raw_capacity),
            active_feature_count,
            output_dtype,
            runtime_name: layout.runtime_name,
            lock_subject: layout.lock_subject,
        }
    }

    pub(crate) fn set_output_dtype(&mut self, value: &str) -> PyResult<()> {
        if self.inner.last_timestamp().is_some() {
            return Err(PyValueError::new_err(format!(
                "output_dtype cannot be changed after the {} has processed an event",
                self.lock_subject
            )));
        }
        self.output_dtype = OutputDtype::parse(value)?;
        Ok(())
    }

    pub(crate) fn symbol(&mut self, name: &str) -> PyResult<usize> {
        let symbol = intern_symbol(name)?;
        if let Some(index) = self
            .symbols
            .iter()
            .position(|candidate| *candidate == symbol)
        {
            return Ok(index);
        }
        self.symbols.push(symbol);
        Ok(self.symbols.len() - 1)
    }

    fn symbol_at(&self, handle: i64) -> PyResult<Symbol> {
        usize::try_from(handle)
            .ok()
            .and_then(|index| self.symbols.get(index).copied())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown symbol handle {handle}; call {}.symbol(name) first",
                    self.runtime_name
                ))
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_event(
        &self,
        kind: u8,
        symbol: i64,
        timestamp: i64,
        price: Option<f64>,
        volume: Option<f64>,
        side: Option<u8>,
        _bid: Option<f64>,
        _ask: Option<f64>,
    ) -> PyResult<Event> {
        let event = match kind {
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
                side.map(parse_trade_side).transpose()?,
            ),
            KIND_ORDERBOOK => {
                return Err(PyValueError::new_err(
                    "KIND_ORDERBOOK no longer accepts bid/ask scalars; use OrderBookEvent with update_order_book or transform_order_book",
                ));
            }
            KIND_TIME => Event::time(timestamp),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported event kind {other} \
                     (expected 0=price, 1=volume, 2=trade, 3=orderbook, 4=time)"
                )));
            }
        };
        event
            .validate_finite_values()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(event)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update(
        &mut self,
        kind: u8,
        symbol: i64,
        timestamp: i64,
        price: Option<f64>,
        volume: Option<f64>,
        side: Option<u8>,
        bid: Option<f64>,
        ask: Option<f64>,
    ) -> PyResult<()> {
        let event = self.build_event(kind, symbol, timestamp, price, volume, side, bid, ask)?;
        self.inner
            .handle_event(event)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transform<'py>(
        &mut self,
        py: Python<'py>,
        kind: PyReadonlyArray1<'py, u8>,
        symbol: PyReadonlyArray1<'py, i64>,
        timestamp: PyReadonlyArray1<'py, i64>,
        price: Option<PyReadonlyArray1<'py, f64>>,
        volume: Option<PyReadonlyArray1<'py, f64>>,
        side: Option<PyReadonlyArray1<'py, u8>>,
        bid: Option<PyReadonlyArray1<'py, f64>>,
        ask: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Py<PyAny>> {
        let kind = kind.as_slice()?;
        let symbol = symbol.as_slice()?;
        let timestamp = timestamp.as_slice()?;
        let n_rows = kind.len();
        if symbol.len() != n_rows || timestamp.len() != n_rows {
            return Err(PyValueError::new_err(
                "kind, symbol and timestamp must have the same length",
            ));
        }

        let price = column("price", &price, n_rows)?;
        let volume = column("volume", &volume, n_rows)?;
        let side = column("side", &side, n_rows)?;
        let bid = column("bid", &bid, n_rows)?;
        let ask = column("ask", &ask, n_rows)?;

        let mut events = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let event = self
                .build_event(
                    kind[row],
                    symbol[row],
                    timestamp[row],
                    price.map(|values| values[row]),
                    volume.map(|values| values[row]),
                    side.map(|values| values[row]),
                    bid.map(|values| values[row]),
                    ask.map(|values| values[row]),
                )
                .map_err(|error| {
                    PyValueError::new_err(format!("row {row}: {}", error.value(py)))
                })?;
            events.push(event);
        }

        self.replay_events(py, events)
    }

    fn validate_order_book(&self, symbol: Symbol) -> PyResult<()> {
        if !self.inner.has_order_book(symbol) {
            return Err(PyValueError::new_err(
                FimlError::OrderBookNotConfigured { symbol }.to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn update_order_book(&mut self, event: &OrderBookEvent) -> PyResult<()> {
        self.validate_order_book(event.symbol)?;
        self.inner
            .handle_event(event.event())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    pub(crate) fn transform_order_book(
        &mut self,
        py: Python<'_>,
        events: Vec<PyRef<'_, OrderBookEvent>>,
    ) -> PyResult<Py<PyAny>> {
        let mut prepared = Vec::with_capacity(events.len());
        for (row, event) in events.iter().enumerate() {
            self.validate_order_book(event.symbol).map_err(|error| {
                PyValueError::new_err(format!("row {row}: {}", error.value(py)))
            })?;
            prepared.push(event.event());
        }
        self.replay_events(py, prepared)
    }

    fn replay_events(&mut self, py: Python<'_>, events: Vec<Event>) -> PyResult<Py<PyAny>> {
        let n_rows = events.len();
        let mut symbol_timestamps = HashMap::new();
        for (row, event) in events.iter().enumerate() {
            let previous_timestamp = symbol_timestamps
                .entry(event.symbol())
                .or_insert_with(|| self.inner.last_timestamp_for_symbol(event.symbol()));
            if let Some(previous_timestamp) = *previous_timestamp
                && previous_timestamp > event.timestamp()
            {
                let error = FimlError::TimestampOutOfOrder {
                    symbol: event.symbol(),
                    event_kind: event.kind(),
                    timestamp: event.timestamp(),
                    previous_timestamp,
                };
                return Err(PyValueError::new_err(format!("row {row}: {error}")));
            }
            *previous_timestamp = Some(event.timestamp());
        }

        let n_features = self.feature_names.len();
        let mut output = OutputBuffer::new(self.output_dtype, n_rows * n_features);
        for (row, event) in events.into_iter().enumerate() {
            self.inner
                .handle_event(event)
                .map_err(|error| PyValueError::new_err(format!("row {row}: {error}")))?;
            output.write_row(row, n_features, self.inner.values());
        }
        output.into_pyarray(py, n_rows, n_features)
    }
}

pub(crate) fn values_to_pyarray(py: Python<'_>, dtype: OutputDtype, values: &[f64]) -> Py<PyAny> {
    match dtype {
        OutputDtype::Float32 => {
            PyArray1::from_vec(py, values.iter().map(|&value| value as f32).collect())
                .into_any()
                .unbind()
        }
        OutputDtype::Float64 => PyArray1::from_slice(py, values).into_any().unbind(),
    }
}

fn parse_trade_side(side: u8) -> PyResult<TradeSide> {
    match side {
        SIDE_AGGRESSOR_BUY => Ok(TradeSide::AgressorBuy),
        SIDE_AGGRESSOR_SELL => Ok(TradeSide::AgressorSell),
        _ => Err(PyValueError::new_err(format!(
            "invalid `side` {side}; expected SIDE_AGGRESSOR_BUY \
             ({SIDE_AGGRESSOR_BUY}) or SIDE_AGGRESSOR_SELL ({SIDE_AGGRESSOR_SELL})"
        ))),
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
fn column<'a, T: Element>(
    name: &str,
    array: &'a Option<PyReadonlyArray1<'_, T>>,
    n_rows: usize,
) -> PyResult<Option<&'a [T]>> {
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

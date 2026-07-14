//! Python bindings for the `fiml` feature extractor.
//!
//! The bindings deliberately run the *exact* Rust extractor: features are
//! computed by replaying events through [`fiml::FeatureExtractor`]'s dispatch,
//! the same code the live Rust environment uses. Build both sides from the same
//! [`fiml::FeatureSet`] JSON and feed the same events in the same order to get
//! identical output. Indicator state is always `f64`; Python arrays can be
//! returned as `float32` or `float64`.

use std::collections::HashSet;
use std::time::Duration;

use fiml::{
    Event, FeatureDef, FeatureExtractor as CoreFeatureExtractor, FeatureSet as CoreFeatureSet,
    IndicatorFeatures, IndicatorSpec, Symbol, symbols,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Event-kind codes for the columnar `transform`/`update` API. They mirror the
/// extractor's event kinds. Each kind reads only the payload columns it needs
/// (see [`FeatureExtractor::build_event`]); `OrderBook` dispatches fine even
/// though no builtin feature subscribes to it yet (the dispatch is a no-op
/// until one does).
const KIND_PRICE: u8 = 0;
const KIND_VOLUME: u8 = 1;
const KIND_TRADE: u8 = 2;
const KIND_ORDERBOOK: u8 = 3;
const KIND_TIME: u8 = 4;

/// Parse a duration string such as `"500ms"`, `"1s"`, `"5m"` or `"1h"`.
/// `field` names the argument in the error message.
fn parse_duration(field: &str, text: &str) -> PyResult<Duration> {
    let text = text.trim();
    let digits = text.len() - text.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let (number, unit) = text.split_at(digits);
    let value: u64 = number.parse().map_err(|_| invalid_duration(field, text))?;
    let unit_millis: u64 = match unit {
        "ms" => 1,
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        _ => return Err(invalid_duration(field, text)),
    };
    value
        .checked_mul(unit_millis)
        .map(Duration::from_millis)
        .ok_or_else(|| invalid_duration(field, text))
}

fn invalid_duration(field: &str, text: &str) -> PyErr {
    PyValueError::new_err(format!(
        "invalid `{field}` duration {text:?}; use an integer with a unit: \
         \"500ms\", \"1s\", \"5m\", \"1h\""
    ))
}

fn parse_event_kind(field: &str, value: &str) -> PyResult<fiml::EventKind> {
    match value {
        "price" => Ok(fiml::EventKind::Price),
        "volume" => Ok(fiml::EventKind::Volume),
        "trade" => Ok(fiml::EventKind::Trade),
        _ => Err(PyValueError::new_err(format!(
            "invalid `{field}` {value:?}; expected \"price\", \"volume\", or \"trade\""
        ))),
    }
}

/// Parse a fixed-offset timezone into an offset from UTC in milliseconds:
/// `"UTC"`, `"UTC+3"`, `"UTC-05:30"`, `"+02:00"`, `"-7"`. Named IANA zones are
/// intentionally unsupported (the core carries no timezone database); pass the
/// session's fixed UTC offset instead.
fn parse_tz(tz: &str) -> PyResult<i64> {
    let invalid = || {
        PyValueError::new_err(format!(
            "invalid `tz` {tz:?}; use \"UTC\" or a fixed offset like \"UTC+3\" \
             or \"-05:30\" (named zones are not supported: the core has no \
             timezone database)"
        ))
    };
    let rest = tz.trim().strip_prefix("UTC").unwrap_or(tz.trim());
    if rest.is_empty() {
        return Ok(0);
    }
    let (sign, body) = match rest.split_at(1) {
        ("+", body) => (1, body),
        ("-", body) => (-1, body),
        _ => return Err(invalid()),
    };
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    let hours: i64 = hours.parse().map_err(|_| invalid())?;
    let minutes: i64 = minutes.parse().map_err(|_| invalid())?;
    if hours > 14 || minutes > 59 {
        return Err(invalid());
    }
    Ok(sign * (hours * 3_600_000 + minutes * 60_000))
}

/// Declarative feature set: the ordered list of features an extractor produces
/// and the parity contract between Python (batch) and Rust (live). Author it
/// with the fluent builder methods, then either construct a
/// [`FeatureExtractor`] from it or `to_json()` it and save the JSON next to the
/// trained model for Rust serving.
#[pyclass]
#[derive(Default)]
pub struct FeatureSet {
    inner: CoreFeatureSet,
}

impl FeatureSet {
    /// Append one feature definition; `name` overrides the generated default.
    fn push(
        &mut self,
        name: Option<String>,
        default_name: String,
        symbol: &str,
        indicator: IndicatorSpec,
    ) {
        self.inner.features.push(FeatureDef {
            name: name.unwrap_or(default_name),
            symbol: symbol.to_string(),
            indicator,
        });
    }
}

#[pymethods]
impl FeatureSet {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Load a feature set from its JSON form (see `to_json`).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: CoreFeatureSet =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Serialize to the JSON parity artifact shared with Rust serving.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Feature (output column) names in definition order.
    fn feature_names(&self) -> Vec<String> {
        self.inner
            .features
            .iter()
            .map(|feature| feature.name.clone())
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.features.len()
    }

    /// Simple moving average over `period` values from `event_kind`.
    #[pyo3(signature = (symbol, period, name=None, *, event_kind="price"))]
    fn sma<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        period: usize,
        name: Option<String>,
        event_kind: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.push(
            name,
            format!("sma_{period}"),
            symbol,
            IndicatorSpec::Sma {
                period,
                event_kind: parse_event_kind("event_kind", event_kind)?,
            },
        );
        Ok(slf)
    }

    /// Exponential moving average over `period` values from `event_kind`.
    #[pyo3(signature = (symbol, period, name=None, *, event_kind="price"))]
    fn ema<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        period: usize,
        name: Option<String>,
        event_kind: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.push(
            name,
            format!("ema_{period}"),
            symbol,
            IndicatorSpec::Ema {
                period,
                event_kind: parse_event_kind("event_kind", event_kind)?,
            },
        );
        Ok(slf)
    }

    /// Time-bucketed SMA of `symbol` prices: buckets of `aggregation` over a
    /// rolling `window` (duration strings, e.g. `aggregation="1s", window="60s"`).
    #[pyo3(signature = (symbol, aggregation, window, name=None))]
    fn sma_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        window: &str,
        name: Option<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let spec = IndicatorSpec::SmaTimed {
            aggregation: parse_duration("aggregation", aggregation)?,
            window: parse_duration("window", window)?,
        };
        slf.push(
            name,
            format!("sma_timed_{aggregation}_{window}"),
            symbol,
            spec,
        );
        Ok(slf)
    }

    /// Time-bucketed on-balance volume of `symbol` trades over a rolling
    /// `window`, bucketed by `aggregation` (duration strings).
    #[pyo3(signature = (symbol, aggregation, window, name=None))]
    fn obv_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        window: &str,
        name: Option<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let spec = IndicatorSpec::ObvTimed {
            aggregation: parse_duration("aggregation", aggregation)?,
            window: parse_duration("window", window)?,
        };
        slf.push(
            name,
            format!("obv_timed_{aggregation}_{window}"),
            symbol,
            spec,
        );
        Ok(slf)
    }

    /// Rolling count of `symbol` trades over a `window`, bucketed by
    /// `aggregation` (duration strings).
    #[pyo3(signature = (symbol, aggregation, window, name=None))]
    fn trade_count_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        window: &str,
        name: Option<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let spec = IndicatorSpec::TradeCountTimed {
            aggregation: parse_duration("aggregation", aggregation)?,
            window: parse_duration("window", window)?,
        };
        slf.push(
            name,
            format!("trade_count_timed_{aggregation}_{window}"),
            symbol,
            spec,
        );
        Ok(slf)
    }

    /// Day-of-week clock feature (`0 = Sunday ..= 6 = Saturday`). Refreshes
    /// from every event's timestamp, so it has a value on every row.
    #[pyo3(signature = (symbol, name=None))]
    fn day_of_week<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        name: Option<String>,
    ) -> PyRefMut<'py, Self> {
        slf.push(
            name,
            "day_of_week".to_string(),
            symbol,
            IndicatorSpec::DayOfWeek,
        );
        slf
    }

    /// Milliseconds since the session opened (the first event after a day
    /// boundary in `tz`). `tz` is `"UTC"` or a fixed offset like `"UTC+3"` /
    /// `"-05:30"`. Refreshes on every event.
    #[pyo3(signature = (symbol, tz="UTC", name=None))]
    fn time_since_session_open<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        tz: &str,
        name: Option<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let spec = IndicatorSpec::TimeSinceSessionOpen {
            utc_offset_millis: parse_tz(tz)?,
        };
        slf.push(name, "time_since_session_open".to_string(), symbol, spec);
        Ok(slf)
    }
}

#[derive(Clone, Copy)]
enum OutputDtype {
    Float32,
    Float64,
}

impl OutputDtype {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            _ => Err(Self::invalid()),
        }
    }

    fn name(self) -> &'static str {
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

fn build_core(feature_set: &CoreFeatureSet) -> PyResult<CoreFeatureExtractor> {
    let mut names = HashSet::with_capacity(feature_set.features.len());
    for feature in &feature_set.features {
        if !names.insert(feature.name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "duplicate feature name {:?}",
                feature.name
            )));
        }
    }
    CoreFeatureExtractor::from_feature_set(feature_set)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A configured, runnable feature extractor.
#[pyclass(subclass)]
pub struct FeatureExtractor {
    inner: CoreFeatureExtractor,
    /// Handle (index) -> interned symbol, so Python can pass cheap integer ids
    /// in array columns instead of strings per row.
    symbols: Vec<Symbol>,
    n_features: usize,
    output_dtype: OutputDtype,
    /// Timestamp of the last event accepted through any mutating Python API.
    last_timestamp: Option<i64>,
}

impl FeatureExtractor {
    fn from_core(inner: CoreFeatureExtractor, output_dtype: OutputDtype) -> Self {
        let n_features = inner.feature_names().len();
        Self {
            inner,
            symbols: Vec::new(),
            n_features,
            output_dtype,
            last_timestamp: None,
        }
    }

    fn validate_global_timestamp(&self, timestamp: i64) -> PyResult<()> {
        if let Some(previous) = self.last_timestamp
            && timestamp < previous
        {
            return Err(PyValueError::new_err(format!(
                "timestamp {timestamp} is before the global timestamp watermark {previous}"
            )));
        }
        Ok(())
    }

    fn symbol_at(&self, handle: i64) -> PyResult<Symbol> {
        usize::try_from(handle)
            .ok()
            .and_then(|index| self.symbols.get(index).copied())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown symbol handle {handle}; call FeatureExtractor.symbol(name) first"
                ))
            })
    }

    /// Build an [`Event`] from the row's event kind and whichever payload
    /// columns it needs. Each kind reads only its required columns; a missing
    /// required column is a `ValueError`. This is the single source of truth for
    /// the kind -> field mapping shared by [`update`](Self::update) (scalars) and
    /// [`transform`](Self::transform) (one row of its columns).
    // The argument list mirrors the extractor's event payload fields and the
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
impl FeatureExtractor {
    /// Build an extractor directly from a [`FeatureSet`].
    #[new]
    #[pyo3(signature = (feature_set, output_dtype="float64"))]
    fn new(feature_set: PyRef<'_, FeatureSet>, output_dtype: &str) -> PyResult<Self> {
        Ok(Self::from_core(
            build_core(&feature_set.inner)?,
            OutputDtype::parse(output_dtype)?,
        ))
    }

    /// Build an extractor from a `FeatureSet` JSON string (the parity contract
    /// shared with the live Rust environment).
    #[staticmethod]
    #[pyo3(signature = (json, output_dtype="float64"))]
    fn from_json(json: &str, output_dtype: &str) -> PyResult<Self> {
        let feature_set: CoreFeatureSet =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_core(
            build_core(&feature_set)?,
            OutputDtype::parse(output_dtype)?,
        ))
    }

    /// Numeric dtype used by arrays returned to Python.
    #[getter]
    fn output_dtype(&self) -> &'static str {
        self.output_dtype.name()
    }

    /// Change the output dtype before the first event is processed.
    #[setter]
    fn set_output_dtype(&mut self, value: &str) -> PyResult<()> {
        if self.last_timestamp.is_some() {
            return Err(PyValueError::new_err(
                "output_dtype cannot be changed after the extractor has processed an event",
            ));
        }
        self.output_dtype = OutputDtype::parse(value)?;
        Ok(())
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

    /// Current feature values in output order. A cell is NaN until its feature
    /// has produced a first value (warmup).
    fn values(&self, py: Python<'_>) -> Py<PyAny> {
        match self.output_dtype {
            OutputDtype::Float32 => {
                let values: Vec<f32> = self
                    .inner
                    .values()
                    .iter()
                    .map(|&value| value as f32)
                    .collect();
                PyArray1::from_vec(py, values).into_any().unbind()
            }
            OutputDtype::Float64 => PyArray1::from_slice(py, self.inner.values())
                .into_any()
                .unbind(),
        }
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
        self.validate_global_timestamp(timestamp)?;
        self.inner
            .validate_dispatch(&event)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .dispatch(&event)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.last_timestamp = Some(timestamp);
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
    /// the length of `kind`. Every row is validated **before** the first
    /// dispatch, so a bad row raises without mutating extractor state. Row `i`
    /// builds its event, dispatches it, then snapshots every feature into row
    /// `i` of the returned `(n_rows, n_features)` matrix in `output_dtype`
    /// (cells are NaN until their feature warms up). Looping in Rust keeps this
    /// fast while using the exact live dispatch path.
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

        // Resolve each optional payload column to a slice once, validating its
        // length here so the per-row hot loop only indexes (no allocation).
        let price = column("price", &price, n_rows)?;
        let volume = column("volume", &volume, n_rows)?;
        let bid = column("bid", &bid, n_rows)?;
        let ask = column("ask", &ask, n_rows)?;

        // All-or-nothing: build and validate every event, including ordering
        // against both extractor state and earlier rows in this batch, before
        // dispatching any event or allocating the output matrix.
        let mut events = Vec::with_capacity(n_rows);
        let mut previous_timestamp = self.last_timestamp;
        for row in 0..n_rows {
            let event = self
                .build_event(
                    kind[row],
                    symbol[row],
                    timestamp[row],
                    price.map(|column| column[row]),
                    volume.map(|column| column[row]),
                    bid.map(|column| column[row]),
                    ask.map(|column| column[row]),
                )
                .map_err(|err| PyValueError::new_err(format!("row {row}: {}", err.value(py))))?;
            if let Some(previous) = previous_timestamp
                && event.timestamp() < previous
            {
                return Err(PyValueError::new_err(format!(
                    "row {row}: timestamp {} is before the global timestamp watermark {previous}",
                    event.timestamp()
                )));
            }
            self.inner
                .validate_dispatch(&event)
                .map_err(|err| PyValueError::new_err(format!("row {row}: {err}")))?;
            previous_timestamp = Some(event.timestamp());
            events.push(event);
        }

        let n_features = self.n_features;
        let mut output = OutputBuffer::new(self.output_dtype, n_rows * n_features);
        for (row, event) in events.iter().enumerate() {
            self.inner
                .dispatch(event)
                .map_err(|err| PyValueError::new_err(format!("row {row}: {err}")))?;
            output.write_row(row, n_features, self.inner.values());
            self.last_timestamp = Some(event.timestamp());
        }
        output.into_pyarray(py, n_rows, n_features)
    }
}

#[pymodule]
fn _fiml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FeatureSet>()?;
    m.add_class::<FeatureExtractor>()?;
    m.add("KIND_PRICE", KIND_PRICE)?;
    m.add("KIND_VOLUME", KIND_VOLUME)?;
    m.add("KIND_TRADE", KIND_TRADE)?;
    m.add("KIND_ORDERBOOK", KIND_ORDERBOOK)?;
    m.add("KIND_TIME", KIND_TIME)?;
    Ok(())
}

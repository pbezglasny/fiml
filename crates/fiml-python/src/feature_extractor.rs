//! Exposes the runnable raw-feature extractor to Python.

use fiml::FeatureVector;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::feature_extractor_spec::FeatureExtractorSpec;
use crate::order_book::OrderBookEvent;
use crate::runtime::{
    CoreFeatureExtractor, EventRuntime, OutputDtype, RuntimeDriver, RuntimeLayout, build_core,
    values_to_pyarray,
};

/// A configured, runnable feature extractor.
#[pyclass(subclass)]
pub struct FeatureExtractor {
    driver: RuntimeDriver<CoreFeatureExtractor>,
}

impl FeatureExtractor {
    fn from_core(inner: CoreFeatureExtractor, output_dtype: OutputDtype) -> Self {
        let active_ids = inner
            .feature_ids()
            .iter()
            .map(|id| id.as_str().to_owned())
            .collect::<Vec<_>>();
        let capacity = inner.feature_vector().capacity();
        Self {
            driver: RuntimeDriver::new(
                inner,
                output_dtype,
                RuntimeLayout {
                    active_ids: active_ids.clone(),
                    capacity,
                    raw_active_ids: active_ids,
                    raw_capacity: capacity,
                    runtime_name: "FeatureExtractor",
                    lock_subject: "extractor",
                },
            ),
        }
    }
}

#[pymethods]
impl FeatureExtractor {
    /// Applies a validated snapshot/delta, preserving core synchronization semantics.
    fn update_order_book(&mut self, event: PyRef<'_, OrderBookEvent>) -> PyResult<()> {
        self.driver.update_order_book(&event)
    }

    /// Prevalidates symbols/timestamps, then replays book events sequentially.
    /// A state-dependent error reports row N; prior rows remain applied and later
    /// rows are skipped. The rejected row retains core resynchronization effects.
    fn transform_order_book(
        &mut self,
        py: Python<'_>,
        events: Vec<PyRef<'_, OrderBookEvent>>,
    ) -> PyResult<Py<PyAny>> {
        self.driver.transform_order_book(py, events)
    }

    /// Build an extractor directly from a [`FeatureExtractorSpec`].
    #[new]
    #[pyo3(signature = (feature_extractor_spec, output_dtype="float64"))]
    fn new(
        feature_extractor_spec: PyRef<'_, FeatureExtractorSpec>,
        output_dtype: &str,
    ) -> PyResult<Self> {
        Ok(Self::from_core(
            build_core(&feature_extractor_spec)?,
            OutputDtype::parse(output_dtype)?,
        ))
    }

    /// Build an extractor directly from versioned FeatureExtractorSpec JSON.
    #[staticmethod]
    #[pyo3(signature = (json, output_dtype="float64"))]
    fn from_json(json: &str, output_dtype: &str) -> PyResult<Self> {
        let feature_extractor_spec = FeatureExtractorSpec::from_json(json)?;
        Ok(Self::from_core(
            build_core(&feature_extractor_spec)?,
            OutputDtype::parse(output_dtype)?,
        ))
    }

    /// Numeric dtype used by arrays returned to Python.
    #[getter]
    fn output_dtype(&self) -> &'static str {
        self.driver.output_dtype.name()
    }

    /// Change the output dtype before the first event is processed.
    #[setter]
    fn set_output_dtype(&mut self, value: &str) -> PyResult<()> {
        self.driver.set_output_dtype(value)
    }

    /// Intern `name` and return a stable integer handle to use in the `symbol`
    /// column of [`transform`](Self::transform) / [`update`](Self::update).
    /// A new name beyond the process-wide 512-symbol limit (including GLOBAL)
    /// raises `ValueError`. Existing names remain usable and handles are unchanged.
    fn symbol(&mut self, name: &str) -> PyResult<usize> {
        self.driver.symbol(name)
    }

    /// Feature (column) names in output order.
    fn feature_names(&self) -> Vec<String> {
        self.driver.feature_names.clone()
    }

    /// Number of feature columns.
    fn n_features(&self) -> usize {
        self.driver.feature_names.len()
    }

    /// Number of configured outputs, excluding trailing reserved cells.
    fn active_feature_count(&self) -> usize {
        self.driver.active_feature_count
    }

    /// Current feature values in output order. A window cell is NaN until its
    /// configured warm-up policy is satisfied and a current value exists.
    fn values(&self, py: Python<'_>) -> Py<PyAny> {
        values_to_pyarray(py, self.driver.output_dtype, self.driver.inner.values())
    }

    /// Apply a single event and update the feature vector. Useful for live
    /// stepping and for checking parity against [`transform`](Self::transform).
    /// Non-finite price or volume payloads raise `ValueError` before changing
    /// feature values, timed state, or the timestamp watermark, regardless of
    /// symbol subscriptions. Finite zero and negative values are accepted.
    ///
    /// Pass only the payload values the event kind needs (see
    /// [`transform`](Self::transform) for the per-kind columns): e.g.
    /// `update(KIND_PRICE, sym, ts, price=...)` or
    /// `update_order_book(OrderBookEvent.snapshot(...))`.
    #[pyo3(signature = (
        kind,
        symbol,
        timestamp,
        *,
        price=None,
        volume=None,
        side=None,
        bid=None,
        ask=None
    ))]
    #[allow(clippy::too_many_arguments)] // payload columns are the Python keyword API
    fn update(
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
        self.driver
            .update(kind, symbol, timestamp, price, volume, side, bid, ask)
    }

    /// Replay a full event stream and return one feature row per input row.
    ///
    /// `kind`, `symbol` and `timestamp` are required and equal length; the
    /// payload columns are optional and each row reads only the columns its kind
    /// needs:
    ///
    /// - `KIND_PRICE` -> `price`
    /// - `KIND_VOLUME` -> `volume`
    /// - `KIND_TRADE` -> `price`, `volume`, and optional `side`
    /// - `KIND_ORDERBOOK` -> rejected; use `transform_order_book`
    /// - `KIND_TIME` -> none
    ///
    /// Required price and volume values must be finite; zero and negative values
    /// are accepted. Unused payload columns are ignored. A non-finite row raises
    /// `ValueError` with `row N:` context before any events in the batch apply.
    ///
    /// A row whose kind needs a column that was not supplied raises a
    /// `ValueError` naming that column. Any provided payload column must match
    /// the length of `kind`. Every row is validated **before** the first
    /// dispatch, so a bad row raises without mutating extractor state. Row `i`
    /// builds its event, dispatches it, then snapshots every feature into row
    /// `i` of the returned `(n_rows, n_features)` matrix in `output_dtype`
    /// (cells are NaN until their feature warms up). Looping in Rust keeps this
    /// fast while using the exact live dispatch path.
    #[pyo3(signature = (
        kind,
        symbol,
        timestamp,
        *,
        price=None,
        volume=None,
        side=None,
        bid=None,
        ask=None
    ))]
    #[allow(clippy::too_many_arguments)] // payload columns are the Python keyword API
    fn transform<'py>(
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
        self.driver
            .transform(py, kind, symbol, timestamp, price, volume, side, bid, ask)
    }
}

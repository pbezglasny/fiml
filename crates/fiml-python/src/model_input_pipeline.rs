//! Exposes the runnable transformed-model pipeline to Python.

use fiml::{PipelineSpec as CorePipelineSpec, VecFeatureVector};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::order_book::OrderBookEvent;
use crate::pipeline_spec::{PipelineSpec, core_feature_ids, model_output_ids};
use crate::runtime::{CorePipeline, OutputDtype, RuntimeDriver, RuntimeLayout, values_to_pyarray};

/// Stateful raw-feature extraction plus fitted model-input transformations.
#[pyclass(subclass)]
pub struct ModelInputPipeline {
    driver: RuntimeDriver<CorePipeline>,
}

impl ModelInputPipeline {
    fn from_spec(spec: &CorePipelineSpec, output_dtype: OutputDtype) -> PyResult<Self> {
        let raw_spec = spec.raw_feature_extractor_spec();
        let raw_vector = VecFeatureVector::new_of_length(
            raw_spec.feature_vector_capacity(),
            raw_spec.feature_vector_length(),
        );
        let model_vector = VecFeatureVector::new_of_length(
            spec.feature_vector_capacity(),
            spec.feature_vector_length(),
        );
        let final_ids = model_output_ids(spec);
        let raw_ids = core_feature_ids(raw_spec);
        let inner = spec
            .build(raw_vector, model_vector)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            driver: RuntimeDriver::new(
                inner,
                output_dtype,
                RuntimeLayout {
                    active_ids: final_ids,
                    capacity: spec.feature_vector_capacity(),
                    raw_active_ids: raw_ids,
                    raw_capacity: raw_spec.feature_vector_capacity(),
                    runtime_name: "ModelInputPipeline",
                    lock_subject: "pipeline",
                },
            ),
        })
    }
}

#[pymethods]
impl ModelInputPipeline {
    /// Whether accepted events have advanced this runtime, for Python recipe locking.
    #[getter]
    fn _has_events(&self) -> bool {
        self.driver.inner.last_timestamp().is_some()
    }

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

    /// Compile a validated model-input spec into an independent runtime.
    #[new]
    #[pyo3(signature = (pipeline_spec, output_dtype="float64"))]
    fn new(pipeline_spec: PyRef<'_, PipelineSpec>, output_dtype: &str) -> PyResult<Self> {
        Self::from_spec(&pipeline_spec.core, OutputDtype::parse(output_dtype)?)
    }

    /// Compile directly from strict canonical model-input JSON.
    #[staticmethod]
    #[pyo3(signature = (json, output_dtype="float64"))]
    fn from_json(json: &str, output_dtype: &str) -> PyResult<Self> {
        let spec: CorePipelineSpec =
            serde_json::from_str(json).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Self::from_spec(&spec, OutputDtype::parse(output_dtype)?)
    }

    /// Numeric dtype used by final and raw arrays returned to Python.
    #[getter]
    fn output_dtype(&self) -> &'static str {
        self.driver.output_dtype.name()
    }

    /// Change the output dtype before the first event is processed.
    #[setter]
    fn set_output_dtype(&mut self, value: &str) -> PyResult<()> {
        self.driver.set_output_dtype(value)
    }

    /// Intern a symbol and return its runtime-local integer handle.
    /// A new name beyond the process-wide 512-symbol limit (including GLOBAL)
    /// raises `ValueError`. Existing names remain usable and handles are unchanged.
    fn symbol(&mut self, name: &str) -> PyResult<usize> {
        self.driver.symbol(name)
    }

    /// Final model-input names, including reserved cells.
    fn feature_names(&self) -> Vec<String> {
        self.driver.feature_names.clone()
    }

    /// Raw diagnostic names, including raw reserved cells.
    fn raw_feature_names(&self) -> Vec<String> {
        self.driver.raw_feature_names.clone()
    }

    /// Complete final width, including reserved cells.
    fn n_features(&self) -> usize {
        self.driver.feature_names.len()
    }

    /// Active final output count, excluding reserved cells.
    fn active_feature_count(&self) -> usize {
        self.driver.active_feature_count
    }

    /// Current final transformed snapshot.
    fn values(&self, py: Python<'_>) -> Py<PyAny> {
        values_to_pyarray(py, self.driver.output_dtype, self.driver.inner.values())
    }

    /// Current raw feature snapshot for diagnostics.
    fn raw_values(&self, py: Python<'_>) -> Py<PyAny> {
        values_to_pyarray(py, self.driver.output_dtype, self.driver.inner.raw_values())
    }

    /// Apply one event and refresh both raw and final snapshots.
    /// Non-finite price or volume payloads raise `ValueError` before changing
    /// raw or final values, timed state, or the timestamp watermark, regardless
    /// of subscriptions. Finite zero and negative values are accepted.
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
    #[allow(clippy::too_many_arguments)]
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

    /// Replay an event stream and return one final model row per event.
    /// Required price and volume values must be finite; zero and negative values
    /// are accepted and unused payload columns are ignored. A non-finite row
    /// raises `ValueError` with `row N:` context before any batch events apply,
    /// leaving raw/final values, timed state, and the timestamp watermark unchanged.
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
    #[allow(clippy::too_many_arguments)]
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

//! Python bindings for the `fiml` feature extractor.
//!
//! The bindings deliberately run the *exact* Rust extractor: features are
//! computed by replaying events through [`fiml::FeatureExtractor`]'s dispatch,
//! the same code the live Rust environment uses. Feed both sides the same
//! feature-vector spec and events in the same order to get identical output. Indicator
//! state is always `f64`; Python arrays can be returned as `float32` or `float64`.

use std::time::Duration;

use fiml::order_book::OrderBookDelta;
use fiml::{
    Event, EventField, EventKind, FeatureDefinition, FeatureExtractor as RustFeatureExtractor,
    FeatureId, FeatureKey, FeatureSource, FeatureVector,
    FeatureVectorSpec as CoreFeatureVectorSpec, FimlError, ModelInputSpec as CoreModelInputSpec,
    Pipeline as RustPipeline, Symbol, TradeSide, TransformationDefinition, VecFeatureVector,
    WarmupPolicy as CoreWarmupPolicy, symbols,
};
use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1};
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

/// Trade-side codes for optional `side` payloads on trade events.
const SIDE_AGGRESSOR_BUY: u8 = 0;
const SIDE_AGGRESSOR_SELL: u8 = 1;

/// Window-indicator warm-up behavior.
#[pyclass(
    name = "WarmupPolicy",
    eq,
    eq_int,
    frozen,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyWarmupPolicy {
    FirstValue,
    FullWindow,
}

impl From<PyWarmupPolicy> for CoreWarmupPolicy {
    fn from(value: PyWarmupPolicy) -> Self {
        match value {
            PyWarmupPolicy::FirstValue => Self::FirstValue,
            PyWarmupPolicy::FullWindow => Self::FullWindow,
        }
    }
}

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

fn parse_value_source(field: &str, value: &str) -> PyResult<EventField> {
    match value {
        "price" => Ok(EventField::Price),
        "volume" => Ok(EventField::Volume),
        "trade_price" => Ok(EventField::TradePrice),
        "trade_volume" => Ok(EventField::TradeVolume),
        _ => Err(PyValueError::new_err(format!(
            "invalid `{field}` {value:?}; expected \"price\", \"volume\", \
             \"trade_price\", or \"trade_volume\""
        ))),
    }
}

fn parse_durations(field: &str, values: Vec<String>) -> PyResult<Vec<Duration>> {
    values
        .iter()
        .map(|value| parse_duration(field, value))
        .collect()
}

/// Parse a fixed-offset timezone into an offset from UTC in milliseconds:
/// `"UTC"`, `"UTC+3"`, `"UTC-05:30"`, `"+02:00"`, `"-7"`. Named IANA zones are
/// intentionally unsupported (the core carries no timezone database); pass a
/// fixed UTC offset instead.
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
    let (sign, body) = if let Some(body) = rest.strip_prefix('+') {
        (1, body)
    } else if let Some(body) = rest.strip_prefix('-') {
        (-1, body)
    } else {
        return Err(invalid());
    };
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    let hours: i64 = hours.parse().map_err(|_| invalid())?;
    let minutes: i64 = minutes.parse().map_err(|_| invalid())?;
    if hours < 0 || minutes < 0 || hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return Err(invalid());
    }
    Ok(sign * (hours * 3_600_000 + minutes * 60_000))
}

fn core_feature_ids(spec: &CoreFeatureVectorSpec) -> Vec<String> {
    spec.definitions()
        .iter()
        .map(|definition| definition.id.as_str().to_owned())
        .collect()
}

fn model_output_ids(spec: &CoreModelInputSpec) -> Vec<String> {
    spec.transformation_definitions()
        .iter()
        .map(|definition| match definition {
            TransformationDefinition::Identity { output, .. }
            | TransformationDefinition::StandardScale { output, .. } => output.as_str().to_owned(),
        })
        .collect()
}

/// Declarative feature-vector spec: the ordered list of features an extractor produces
/// and the parity contract between Python (batch) and Rust (live). Author it
/// with the fluent builder methods, then construct a [`FeatureExtractor`] from
/// it.
#[pyclass]
pub struct FeatureVectorSpec {
    core: CoreFeatureVectorSpec,
    explicit_capacity: bool,
}

impl FeatureVectorSpec {
    fn add_group<I>(&mut self, definitions: I) -> PyResult<()>
    where
        I: IntoIterator<Item = FeatureDefinition>,
    {
        let mut all_definitions = self.core.definitions().to_vec();
        all_definitions.extend(definitions);
        let capacity = if self.explicit_capacity {
            self.core.feature_vector_capacity()
        } else {
            all_definitions.len()
        };
        self.core = CoreFeatureVectorSpec::with_metadata(
            all_definitions,
            capacity,
            self.core.checksum().map(str::to_owned),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(())
    }

    fn require_windows<T>(windows: &[T]) -> PyResult<()> {
        if windows.is_empty() {
            Err(PyValueError::new_err("windows must not be empty"))
        } else {
            Ok(())
        }
    }
}

#[pymethods]
impl FeatureVectorSpec {
    #[new]
    #[pyo3(signature = (*, capacity=None, checksum=None))]
    fn new(capacity: Option<usize>, checksum: Option<String>) -> PyResult<Self> {
        let explicit_capacity = capacity.is_some();
        let core = CoreFeatureVectorSpec::with_metadata([], capacity.unwrap_or(0), checksum)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            core,
            explicit_capacity,
        })
    }

    /// Loads the strict versioned JSON parity artifact.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let core =
            serde_json::from_str(json).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            core,
            explicit_capacity: true,
        })
    }

    /// Serializes this spec using the canonical Rust JSON adapter.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.core)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Number of grouped builder calls. Compatible calls may share one runtime
    /// derivation after compilation.
    fn indicator_count(&self) -> usize {
        self.core.indicator_count()
    }

    /// Number of output cells produced after compilation.
    fn output_count(&self) -> usize {
        self.core.feature_vector_length()
    }

    /// Complete configured model width, including trailing reserved cells.
    #[getter]
    fn capacity(&self) -> usize {
        self.core.feature_vector_capacity()
    }

    /// Number of configured scalar outputs, excluding reserved cells.
    #[getter]
    fn active_feature_count(&self) -> usize {
        self.core.feature_vector_length()
    }

    /// Opaque checksum metadata from the parity artifact.
    #[getter]
    fn checksum(&self) -> Option<&str> {
        self.core.checksum()
    }

    /// Active stable feature IDs in canonical raw-vector order.
    fn feature_ids(&self) -> Vec<String> {
        core_feature_ids(&self.core)
    }

    /// Grouped simple moving averages over ordered sample windows.
    #[pyo3(signature = (
        symbol,
        windows,
        *,
        source="price",
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn sma<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        windows: Vec<usize>,
        source: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = symbols::intern(symbol);
        let field = parse_value_source("source", source)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::Sma {
                symbol,
                source: FeatureSource::Field(field),
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped exponential moving averages over ordered sample windows.
    #[pyo3(signature = (
        symbol,
        windows,
        *,
        source="price",
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn ema<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        windows: Vec<usize>,
        source: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = symbols::intern(symbol);
        let field = parse_value_source("source", source)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::Ema {
                symbol,
                source: FeatureSource::Field(field),
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped cumulative-volume-delta windows over classified trades.
    #[pyo3(signature = (
        symbol,
        windows,
        *,
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn cvd<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        windows: Vec<usize>,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = symbols::intern(symbol);
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::Cvd {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped time-bucketed moving averages over ordered duration windows.
    #[pyo3(signature = (
        symbol,
        aggregation,
        windows,
        *,
        source="price",
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn sma_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        windows: Vec<String>,
        source: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = symbols::intern(symbol);
        let field = parse_value_source("source", source)?;
        let aggregation = parse_duration("aggregation", aggregation)?;
        let windows = parse_durations("windows", windows)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::SmaTimed {
                symbol,
                source: FeatureSource::Field(field),
                aggregation,
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped time-bucketed on-balance-volume windows.
    #[pyo3(signature = (
        symbol,
        aggregation,
        windows,
        *,
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn obv_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        windows: Vec<String>,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = symbols::intern(symbol);
        let aggregation = parse_duration("aggregation", aggregation)?;
        let windows = parse_durations("windows", windows)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::ObvTimed {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                aggregation,
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Rolling count of `symbol` trades over a `window`, bucketed by
    /// `aggregation` (duration strings).
    #[pyo3(signature = (
        symbol,
        aggregation,
        window,
        *,
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn trade_count_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        window: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let symbol = symbols::intern(symbol);
        let aggregation = parse_duration("aggregation", aggregation)?;
        let window = parse_duration("window", window)?;
        slf.add_group([definition(FeatureKey::TradeCountTimed {
            symbol,
            source: FeatureSource::Event(EventKind::Trade),
            aggregation,
            window,
            warmup_policy: warmup.into(),
        })])?;
        Ok(slf)
    }

    /// Day-of-week clock feature (`0 = Sunday ..= 6 = Saturday`). Refreshes
    /// from every event's timestamp, so it has a value on every row.
    fn day_of_week(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.add_group([definition(FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::EveryEvent,
        })])?;
        Ok(slf)
    }

    /// Milliseconds since the first observed event after a local day boundary.
    #[pyo3(signature = (tz="UTC"))]
    fn time_since_first_event_of_day<'py>(
        mut slf: PyRefMut<'py, Self>,
        tz: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let utc_offset_millis = parse_tz(tz)?;
        slf.add_group([definition(FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::EveryEvent,
            utc_offset_millis,
        })])?;
        Ok(slf)
    }
}

/// Validated raw-feature and fitted scalar-transformation specification.
///
/// The raw spec is cloned at construction so later Python builder mutations do
/// not change the model artifact. Transformations remain in authored order.
#[pyclass]
pub struct ModelInputSpec {
    core: CoreModelInputSpec,
    explicit_capacity: bool,
}

impl ModelInputSpec {
    fn add_transformation(&mut self, definition: TransformationDefinition) -> PyResult<()> {
        let mut definitions = self.core.transformation_definitions().to_vec();
        definitions.push(definition);
        let capacity = if self.explicit_capacity {
            self.core.feature_vector_capacity()
        } else {
            definitions.len()
        };
        let candidate = CoreModelInputSpec::with_metadata(
            self.core.raw_feature_vector_spec().clone(),
            definitions,
            capacity,
            self.core.checksum().map(str::to_owned),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.core = candidate;
        Ok(())
    }
}

#[pymethods]
impl ModelInputSpec {
    #[new]
    #[pyo3(signature = (raw_feature_vector_spec, *, capacity=None, checksum=None))]
    fn new(
        raw_feature_vector_spec: PyRef<'_, FeatureVectorSpec>,
        capacity: Option<usize>,
        checksum: Option<String>,
    ) -> PyResult<Self> {
        let explicit_capacity = capacity.is_some();
        let core = CoreModelInputSpec::with_metadata(
            raw_feature_vector_spec.core.clone(),
            [],
            capacity.unwrap_or(0),
            checksum,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            core,
            explicit_capacity,
        })
    }

    /// Loads the strict versioned canonical model-input artifact.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let core =
            serde_json::from_str(json).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            core,
            explicit_capacity: true,
        })
    }

    /// Serializes this spec using the canonical Rust JSON adapter.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.core)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Append an unchanged scalar to the final model vector.
    #[pyo3(signature = (input, *, output=None))]
    fn identity<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let output = output.unwrap_or(input);
        slf.add_transformation(TransformationDefinition::identity(
            FeatureId::new(input),
            FeatureId::new(output),
        ))?;
        Ok(slf)
    }

    /// Append a fitted `(input - mean) / scale` scalar transformation.
    #[pyo3(signature = (input, *, mean, scale, output=None))]
    fn standard_scale<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        mean: f64,
        scale: f64,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let output = output.unwrap_or(input);
        slf.add_transformation(TransformationDefinition::standard_scale(
            FeatureId::new(input),
            FeatureId::new(output),
            mean,
            scale,
        ))?;
        Ok(slf)
    }

    /// Active final IDs in authored transformation order.
    fn feature_ids(&self) -> Vec<String> {
        model_output_ids(&self.core)
    }

    /// Active raw IDs in canonical extraction order.
    fn raw_feature_ids(&self) -> Vec<String> {
        core_feature_ids(self.core.raw_feature_vector_spec())
    }

    /// Complete configured final width, including trailing reserved cells.
    #[getter]
    fn capacity(&self) -> usize {
        self.core.feature_vector_capacity()
    }

    /// Number of configured final outputs, excluding reserved cells.
    #[getter]
    fn active_feature_count(&self) -> usize {
        self.core.feature_vector_length()
    }

    /// Opaque model checksum, independent of the raw-spec checksum.
    #[getter]
    fn checksum(&self) -> Option<&str> {
        self.core.checksum()
    }
}

fn definition(key: FeatureKey) -> FeatureDefinition {
    FeatureDefinition::with_default_id(key)
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

type CoreFeatureExtractor = RustFeatureExtractor<VecFeatureVector>;
type CorePipeline = RustPipeline<VecFeatureVector, VecFeatureVector>;

fn build_core(feature_vector_spec: &FeatureVectorSpec) -> PyResult<CoreFeatureExtractor> {
    let output_vector = VecFeatureVector::new_of_length(
        feature_vector_spec.core.feature_vector_capacity(),
        feature_vector_spec.core.feature_vector_length(),
    );
    feature_vector_spec
        .core
        .build(output_vector)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn complete_names(active_ids: &[String], capacity: usize) -> Vec<String> {
    let mut names = active_ids.to_vec();
    names.extend((active_ids.len()..capacity).map(|index| format!("__reserved_{index}")));
    names
}

trait EventRuntime {
    fn handle_event(&mut self, event: Event) -> fiml::Result<()>;
    fn last_timestamp(&self) -> Option<i64>;
    fn values(&self) -> &[f64];
}

impl EventRuntime for CoreFeatureExtractor {
    fn handle_event(&mut self, event: Event) -> fiml::Result<()> {
        RustFeatureExtractor::handle_event(self, event).map(|_| ())
    }

    fn last_timestamp(&self) -> Option<i64> {
        RustFeatureExtractor::last_timestamp(self)
    }

    fn values(&self) -> &[f64] {
        self.feature_vector().values()
    }
}

impl EventRuntime for CorePipeline {
    fn handle_event(&mut self, event: Event) -> fiml::Result<()> {
        RustPipeline::handle_event(self, event).map(|_| ())
    }

    fn last_timestamp(&self) -> Option<i64> {
        RustPipeline::last_timestamp(self)
    }

    fn values(&self) -> &[f64] {
        RustPipeline::values(self)
    }
}

struct RuntimeDriver<R>
where
    R: EventRuntime,
{
    inner: R,
    symbols: Vec<Symbol>,
    feature_names: Vec<String>,
    raw_feature_names: Vec<String>,
    active_feature_count: usize,
    output_dtype: OutputDtype,
    runtime_name: &'static str,
    lock_subject: &'static str,
}

struct RuntimeLayout {
    active_ids: Vec<String>,
    capacity: usize,
    raw_active_ids: Vec<String>,
    raw_capacity: usize,
    runtime_name: &'static str,
    lock_subject: &'static str,
}

impl<R> RuntimeDriver<R>
where
    R: EventRuntime,
{
    fn new(inner: R, output_dtype: OutputDtype, layout: RuntimeLayout) -> Self {
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

    fn set_output_dtype(&mut self, value: &str) -> PyResult<()> {
        if self.inner.last_timestamp().is_some() {
            return Err(PyValueError::new_err(format!(
                "output_dtype cannot be changed after the {} has processed an event",
                self.lock_subject
            )));
        }
        self.output_dtype = OutputDtype::parse(value)?;
        Ok(())
    }

    fn symbol(&mut self, name: &str) -> usize {
        let symbol = symbols::intern(name);
        if let Some(index) = self
            .symbols
            .iter()
            .position(|candidate| *candidate == symbol)
        {
            return index;
        }
        self.symbols.push(symbol);
        self.symbols.len() - 1
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
        bid: Option<f64>,
        ask: Option<f64>,
    ) -> PyResult<Event> {
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
                side.map(parse_trade_side).transpose()?,
            ),
            KIND_ORDERBOOK => {
                let _ = require("bid", bid)?;
                let _ = require("ask", ask)?;
                Event::order_book_delta(
                    self.symbol_at(symbol)?,
                    timestamp,
                    OrderBookDelta::new(0, Vec::new()),
                )
            }
            KIND_TIME => Event::time(timestamp),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported event kind {other} \
                     (expected 0=price, 1=volume, 2=trade, 3=orderbook, 4=time)"
                )));
            }
        })
    }

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
        let event = self.build_event(kind, symbol, timestamp, price, volume, side, bid, ask)?;
        self.inner
            .handle_event(event)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

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

        let mut previous_timestamp = self.inner.last_timestamp();
        for (row, event) in events.iter().enumerate() {
            if let Some(previous_timestamp) = previous_timestamp
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
            previous_timestamp = Some(event.timestamp());
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

fn values_to_pyarray(py: Python<'_>, dtype: OutputDtype, values: &[f64]) -> Py<PyAny> {
    match dtype {
        OutputDtype::Float32 => {
            PyArray1::from_vec(py, values.iter().map(|&value| value as f32).collect())
                .into_any()
                .unbind()
        }
        OutputDtype::Float64 => PyArray1::from_slice(py, values).into_any().unbind(),
    }
}

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

#[pymethods]
impl FeatureExtractor {
    /// Build an extractor directly from a [`FeatureVectorSpec`].
    #[new]
    #[pyo3(signature = (feature_vector_spec, output_dtype="float64"))]
    fn new(
        feature_vector_spec: PyRef<'_, FeatureVectorSpec>,
        output_dtype: &str,
    ) -> PyResult<Self> {
        Ok(Self::from_core(
            build_core(&feature_vector_spec)?,
            OutputDtype::parse(output_dtype)?,
        ))
    }

    /// Build an extractor directly from versioned FeatureVectorSpec JSON.
    #[staticmethod]
    #[pyo3(signature = (json, output_dtype="float64"))]
    fn from_json(json: &str, output_dtype: &str) -> PyResult<Self> {
        let feature_vector_spec = FeatureVectorSpec::from_json(json)?;
        Ok(Self::from_core(
            build_core(&feature_vector_spec)?,
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
    fn symbol(&mut self, name: &str) -> usize {
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
    ///
    /// Pass only the payload values the event kind needs (see
    /// [`transform`](Self::transform) for the per-kind columns): e.g.
    /// `update(KIND_PRICE, sym, ts, price=...)` or
    /// `update(KIND_ORDERBOOK, sym, ts, bid=..., ask=...)`.
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

/// Stateful raw-feature extraction plus fitted model-input transformations.
#[pyclass(subclass)]
pub struct ModelInputPipeline {
    driver: RuntimeDriver<CorePipeline>,
}

impl ModelInputPipeline {
    fn from_spec(spec: &CoreModelInputSpec, output_dtype: OutputDtype) -> PyResult<Self> {
        let raw_spec = spec.raw_feature_vector_spec();
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
    /// Compile a validated model-input spec into an independent runtime.
    #[new]
    #[pyo3(signature = (model_input_spec, output_dtype="float64"))]
    fn new(model_input_spec: PyRef<'_, ModelInputSpec>, output_dtype: &str) -> PyResult<Self> {
        Self::from_spec(&model_input_spec.core, OutputDtype::parse(output_dtype)?)
    }

    /// Compile directly from strict canonical model-input JSON.
    #[staticmethod]
    #[pyo3(signature = (json, output_dtype="float64"))]
    fn from_json(json: &str, output_dtype: &str) -> PyResult<Self> {
        let spec: CoreModelInputSpec =
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
    fn symbol(&mut self, name: &str) -> usize {
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

#[pymodule]
fn _fiml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWarmupPolicy>()?;
    m.add_class::<FeatureVectorSpec>()?;
    m.add_class::<ModelInputSpec>()?;
    m.add_class::<FeatureExtractor>()?;
    m.add_class::<ModelInputPipeline>()?;
    m.add("KIND_PRICE", KIND_PRICE)?;
    m.add("KIND_VOLUME", KIND_VOLUME)?;
    m.add("KIND_TRADE", KIND_TRADE)?;
    m.add("KIND_ORDERBOOK", KIND_ORDERBOOK)?;
    m.add("KIND_TIME", KIND_TIME)?;
    m.add("SIDE_AGGRESSOR_BUY", SIDE_AGGRESSOR_BUY)?;
    m.add("SIDE_AGGRESSOR_SELL", SIDE_AGGRESSOR_SELL)?;
    Ok(())
}

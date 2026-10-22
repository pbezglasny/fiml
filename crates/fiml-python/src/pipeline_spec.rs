//! Defines model transformations and the Python-facing pipeline specification.

use std::time::Duration;

use fiml::{
    FeatureExtractorSpec as CoreFeatureExtractorSpec, FeatureId, FittedStage,
    PipelineSpec as CorePipelineSpec, TransformerDefinition, WarmupPolicy,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::feature_extractor_spec::{FeatureExtractorSpec, PyWarmupPolicy, parse_duration};

fn rolling_transformation(
    input: &str,
    window: usize,
    warmup: PyWarmupPolicy,
    output: Option<&str>,
    constructor: fn(FeatureId, FeatureId, usize, WarmupPolicy) -> TransformerDefinition,
) -> TransformerDefinition {
    constructor(
        FeatureId::new(input),
        FeatureId::new(output.unwrap_or(input)),
        window,
        warmup.into(),
    )
}

fn rolling_timed_transformation(
    input: &str,
    window: &str,
    aggregation: &str,
    warmup: PyWarmupPolicy,
    output: Option<&str>,
    constructor: fn(
        FeatureId,
        FeatureId,
        Duration,
        Duration,
        WarmupPolicy,
    ) -> TransformerDefinition,
) -> PyResult<TransformerDefinition> {
    let window = parse_duration("window", window)?;
    let aggregation = parse_duration("aggregation", aggregation)?;
    Ok(constructor(
        FeatureId::new(input),
        FeatureId::new(output.unwrap_or(input)),
        window,
        aggregation,
        warmup.into(),
    ))
}

fn timed_transformation(
    input: &str,
    lag_window: &str,
    aggregation: &str,
    output: Option<&str>,
    constructor: fn(FeatureId, FeatureId, Duration, Duration) -> TransformerDefinition,
) -> PyResult<TransformerDefinition> {
    let window = parse_duration("lag_window", lag_window)?;
    let aggregation = parse_duration("aggregation", aggregation)?;
    Ok(constructor(
        FeatureId::new(input),
        FeatureId::new(output.unwrap_or(input)),
        window,
        aggregation,
    ))
}

/// Authors one group of scalar outputs whose inputs are resolved against the preceding stage.
/// Resolution is deferred until fitting because PCA output IDs may not exist yet.
#[pyclass]
#[derive(Clone, Default)]
pub struct ScalarStage {
    definitions: Vec<TransformerDefinition>,
}

#[pymethods]
impl ScalarStage {
    /// Snapshot numeric definitions for graph insertion; parent validation is deferred.
    fn _to_json(&self) -> PyResult<String> {
        let stage = FittedStage::Scalar {
            transformations: self.definitions.clone(),
        };
        if !self.definitions.is_empty() {
            let inputs = self
                .definitions
                .iter()
                .map(|definition| definition.input.clone())
                .collect::<Vec<_>>();
            stage.validate(&inputs).map_err(PyValueError::new_err)?;
        }
        serde_json::to_string(&stage).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Consecutive observations greater than epsilon.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn positive_streak<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::positive_streak(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            epsilon,
        ));
        slf
    }

    /// Consecutive observations less than negative epsilon.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn negative_streak<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::negative_streak(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            epsilon,
        ));
        slf
    }

    /// Finite observations since the last value greater than epsilon.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn observations_since_positive<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions
            .push(TransformerDefinition::observations_since_positive(
                FeatureId::new(input),
                FeatureId::new(output.unwrap_or(input)),
                epsilon,
            ));
        slf
    }

    /// Finite observations since the last value less than negative epsilon.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn observations_since_negative<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions
            .push(TransformerDefinition::observations_since_negative(
                FeatureId::new(input),
                FeatureId::new(output.unwrap_or(input)),
                epsilon,
            ));
        slf
    }

    /// Finite observations since the last value outside the neutral interval.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn observations_since_nonzero<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions
            .push(TransformerDefinition::observations_since_nonzero(
                FeatureId::new(input),
                FeatureId::new(output.unwrap_or(input)),
                epsilon,
            ));
        slf
    }

    /// Elapsed milliseconds since the last value greater than epsilon.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn time_since_positive<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions
            .push(TransformerDefinition::time_since_positive(
                FeatureId::new(input),
                FeatureId::new(output.unwrap_or(input)),
                epsilon,
            ));
        slf
    }

    /// Elapsed milliseconds since the last value less than negative epsilon.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn time_since_negative<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions
            .push(TransformerDefinition::time_since_negative(
                FeatureId::new(input),
                FeatureId::new(output.unwrap_or(input)),
                epsilon,
            ));
        slf
    }

    /// Elapsed milliseconds since the last value outside the neutral interval.
    #[pyo3(signature = (input, *, epsilon=0., output=None))]
    fn time_since_nonzero<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        epsilon: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions
            .push(TransformerDefinition::time_since_nonzero(
                FeatureId::new(input),
                FeatureId::new(output.unwrap_or(input)),
                epsilon,
            ));
        slf
    }
    /// Least-squares slope against sample indices; NaN with fewer than two observations.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_slope<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_slope,
        ));
        slf
    }
    /// Sum of absolute adjacent changes within the window.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_path_length<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_path_length,
        ));
        slf
    }
    /// Sum of positive adjacent changes within the window.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_positive_movement<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_positive_movement,
        ));
        slf
    }
    /// Magnitude of negative adjacent changes within the window.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_negative_movement<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_negative_movement,
        ));
        slf
    }
    /// Number of reversals between nonzero deltas; zero deltas are ignored.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_direction_change_count<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_direction_change_count,
        ));
        slf
    }
    /// Largest earlier-peak to later-trough decrease, in absolute input units.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_max_drawdown<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_max_drawdown,
        ));
        slf
    }
    /// Rolling population standard deviation of finite observations, with shared history across windows.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_std<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_std,
        ));
        slf
    }
    /// Rolling population standard deviation of finite observations, with shared history across windows.
    /// Timed buckets retain every finite observation and expire together.
    #[pyo3(signature = (input, *, window, aggregation="1ms", warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_std_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: &str,
        aggregation: &str,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(rolling_timed_transformation(
            input,
            window,
            aggregation,
            warmup,
            output,
            TransformerDefinition::rolling_std_timed_with_aggregation,
        )?);
        Ok(slf)
    }
    /// Rolling minimum of finite observations, with shared history across windows.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_min<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_min,
        ));
        slf
    }
    /// Rolling minimum of finite observations, with shared history across windows.
    /// Timed buckets retain every finite observation and expire together.
    #[pyo3(signature = (input, *, window, aggregation="1ms", warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_min_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: &str,
        aggregation: &str,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(rolling_timed_transformation(
            input,
            window,
            aggregation,
            warmup,
            output,
            TransformerDefinition::rolling_min_timed_with_aggregation,
        )?);
        Ok(slf)
    }
    /// Rolling maximum of finite observations, with shared history across windows.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_max<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_max,
        ));
        slf
    }
    /// Rolling maximum of finite observations, with shared history across windows.
    /// Timed buckets retain every finite observation and expire together.
    #[pyo3(signature = (input, *, window, aggregation="1ms", warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_max_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: &str,
        aggregation: &str,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(rolling_timed_transformation(
            input,
            window,
            aggregation,
            warmup,
            output,
            TransformerDefinition::rolling_max_timed_with_aggregation,
        )?);
        Ok(slf)
    }
    /// Rolling range (maximum minus minimum) of finite observations, with shared history across windows.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_range<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(rolling_transformation(
            input,
            window,
            warmup,
            output,
            TransformerDefinition::rolling_range,
        ));
        slf
    }
    /// Rolling range (maximum minus minimum) of finite observations, with shared history across windows.
    /// Timed buckets retain every finite observation and expire together.
    #[pyo3(signature = (input, *, window, aggregation="1ms", warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn rolling_range_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: &str,
        aggregation: &str,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(rolling_timed_transformation(
            input,
            window,
            aggregation,
            warmup,
            output,
            TransformerDefinition::rolling_range_timed_with_aggregation,
        )?);
        Ok(slf)
    }

    /// Apply an as-of operation on the input's source clock, exact at the default 1ms aggregation.
    /// Positive whole-millisecond duration strings include "100ms", "500ms", and "1s".
    /// Aggregation retains the last finite observation per bucket and reduces memory and precision.
    /// Lag must be a positive multiple of aggregation; history is capped at 1,000,000 slots.
    /// Clock advances refresh output; missing observations emit NaN without entering history.
    #[pyo3(signature = (input, *, lag_window, aggregation="1ms", output=None))]
    fn lagged_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: &str,
        aggregation: &str,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(timed_transformation(
            input,
            lag_window,
            aggregation,
            output,
            TransformerDefinition::lagged_timed_with_aggregation,
        )?);
        Ok(slf)
    }

    /// Apply an as-of operation on the input's source clock, exact at the default 1ms aggregation.
    /// Positive whole-millisecond duration strings include "100ms", "500ms", and "1s".
    /// Aggregation retains the last finite observation per bucket and reduces memory and precision.
    /// Lag must be a positive multiple of aggregation; history is capped at 1,000,000 slots.
    /// Clock advances refresh output; missing observations emit NaN without entering history.
    #[pyo3(signature = (input, *, lag_window, aggregation="1ms", output=None))]
    fn delta_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: &str,
        aggregation: &str,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(timed_transformation(
            input,
            lag_window,
            aggregation,
            output,
            TransformerDefinition::delta_timed_with_aggregation,
        )?);
        Ok(slf)
    }

    /// Apply an as-of operation on the input's source clock, exact at the default 1ms aggregation.
    /// Positive whole-millisecond duration strings include "100ms", "500ms", and "1s".
    /// Aggregation retains the last finite observation per bucket and reduces memory and precision.
    /// Lag must be a positive multiple of aggregation; history is capped at 1,000,000 slots.
    /// Clock advances refresh output; missing observations emit NaN without entering history.
    #[pyo3(signature = (input, *, lag_window, aggregation="1ms", output=None))]
    fn simple_return_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: &str,
        aggregation: &str,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(timed_transformation(
            input,
            lag_window,
            aggregation,
            output,
            TransformerDefinition::simple_return_timed_with_aggregation,
        )?);
        Ok(slf)
    }

    /// Apply an as-of operation on the input's source clock, exact at the default 1ms aggregation.
    /// Positive whole-millisecond duration strings include "100ms", "500ms", and "1s".
    /// Aggregation retains the last finite observation per bucket and reduces memory and precision.
    /// Lag must be a positive multiple of aggregation; history is capped at 1,000,000 slots.
    /// Clock advances refresh output; missing observations emit NaN without entering history.
    #[pyo3(signature = (input, *, lag_window, aggregation="1ms", output=None))]
    fn log_return_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: &str,
        aggregation: &str,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.definitions.push(timed_transformation(
            input,
            lag_window,
            aggregation,
            output,
            TransformerDefinition::log_return_timed_with_aggregation,
        )?);
        Ok(slf)
    }

    /// Compute an exact delta over finite input observations.
    /// Lag must be in 1..=10_000; observed nonfinite inputs emit NaN without advancing history.
    /// Unobserved events retain the last output. Exact lag k requires k + 1 finite observations.
    #[pyo3(signature = (input, *, lag_window, output=None))]
    fn delta<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: usize,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::delta(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            lag_window,
        ));
        slf
    }

    /// Compute an exact simple return over finite input observations.
    /// Lag must be in 1..=10_000; observed nonfinite inputs emit NaN without advancing history.
    /// Unobserved events retain the last output. Exact lag k requires k + 1 finite observations.
    #[pyo3(signature = (input, *, lag_window, output=None))]
    fn simple_return<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: usize,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::simple_return(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            lag_window,
        ));
        slf
    }

    /// Compute an exact log return over finite input observations.
    /// Lag must be in 1..=10_000; observed nonfinite inputs emit NaN without advancing history.
    /// Unobserved events retain the last output. Exact lag k requires k + 1 finite observations.
    #[pyo3(signature = (input, *, lag_window, output=None))]
    fn log_return<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: usize,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::log_return(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            lag_window,
        ));
        slf
    }

    /// Average finite source observations using the existing EMA calculator.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn ema<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        let definition = TransformerDefinition::ema(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            window,
            warmup.into(),
        );
        slf.definitions.push(definition);
        slf
    }

    /// Average finite source observations using the existing SMA calculator.
    #[pyo3(signature = (input, *, window, warmup=PyWarmupPolicy::FullWindow, output=None))]
    fn sma<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        window: usize,
        warmup: PyWarmupPolicy,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        let definition = TransformerDefinition::sma(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            window,
            warmup.into(),
        );
        slf.definitions.push(definition);
        slf
    }

    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Independent recipe snapshot, so later edits cannot change a pipeline.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Copy a scalar from the preceding stage.
    #[pyo3(signature = (input, *, output=None))]
    fn identity<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::identity(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
        ));
        slf
    }

    /// Emit a preceding-stage scalar from `lag_window` accepted events earlier.
    #[pyo3(signature = (input, *, lag_window, output=None))]
    fn lagged<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: usize,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::lagged(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            lag_window,
        ));
        slf
    }

    /// Apply fixed `(input - mean) / scale` parameters to a preceding-stage scalar.
    #[pyo3(signature = (input, *, mean, scale, output=None))]
    fn standard_scale<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        mean: f64,
        scale: f64,
        output: Option<&str>,
    ) -> PyRefMut<'py, Self> {
        slf.definitions.push(TransformerDefinition::standard_scale(
            FeatureId::new(input),
            FeatureId::new(output.unwrap_or(input)),
            mean,
            scale,
        ));
        slf
    }
}

pub(crate) fn core_feature_ids(spec: &CoreFeatureExtractorSpec) -> Vec<String> {
    spec.definitions()
        .iter()
        .map(|definition| definition.id.as_str().to_owned())
        .collect()
}

/// Numeric graph adapter used by Python authoring and fitted artifact loading.
#[pyclass]
#[derive(Clone)]
pub struct PipelineSpec {
    pub(crate) core: CorePipelineSpec,
}

#[pymethods]
impl PipelineSpec {
    #[new]
    #[pyo3(signature = (extractor, *, checksum=None))]
    fn new(extractor: PyRef<'_, FeatureExtractorSpec>, checksum: Option<String>) -> Self {
        let mut core = CorePipelineSpec::new(extractor.core.clone());
        core.set_checksum(checksum);
        Self { core }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn _intern_symbol(&self, name: &str) -> PyResult<String> {
        crate::intern_symbol(name).map(|symbol| symbol.resolve_as_string())
    }

    fn add(&mut self, parent: usize, stage: &str) -> PyResult<usize> {
        let parent = self
            .core
            .node(parent)
            .ok_or_else(|| PyValueError::new_err("parent must be an earlier node"))?;
        let stage: FittedStage = serde_json::from_str(stage)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.core
            .add(parent, stage)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(self.core.nodes().len())
    }

    fn output(&mut self, name: &str, node: usize) -> PyResult<()> {
        let node = self
            .core
            .node(node)
            .ok_or_else(|| PyValueError::new_err("output node index out of bounds"))?;
        self.core
            .output(name, node)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn validate(&self) -> PyResult<()> {
        self.core
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn feature_ids(&self, node: usize) -> PyResult<Vec<String>> {
        let node = self
            .core
            .node(node)
            .ok_or_else(|| PyValueError::new_err("node index out of bounds"))?;
        Ok(self
            .core
            .feature_ids(node)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .map(|id| id.as_str().to_owned())
            .collect())
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            core: serde_json::from_str(json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.core)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

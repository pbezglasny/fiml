//! Defines model transformations and the Python-facing pipeline specification.

use fiml::{
    FeatureExtractorSpec as CoreFeatureExtractorSpec, FeatureId, FittedStage,
    PipelineSpec as CorePipelineSpec, TransformerDefinition,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::feature_extractor_spec::FeatureExtractorSpec;

/// Authors one group of scalar outputs whose inputs are resolved against the preceding stage.
/// Resolution is deferred until fitting because PCA output IDs may not exist yet.
#[pyclass]
#[derive(Clone, Default)]
pub struct ScalarStage {
    definitions: Vec<TransformerDefinition>,
}

#[pymethods]
impl ScalarStage {
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

pub(crate) fn model_output_ids(spec: &CorePipelineSpec) -> Vec<String> {
    spec.output_ids()
        .iter()
        .map(|id| id.as_str().to_owned())
        .collect()
}

/// Validated raw-feature and fitted scalar-transformation specification.
///
/// The raw spec is cloned at construction so later Python builder mutations do
/// not change the model artifact. Transformations remain in authored order.
#[pyclass]
#[derive(Clone)]
pub struct PipelineSpec {
    pub(crate) core: CorePipelineSpec,
    explicit_capacity: bool,
}

impl PipelineSpec {
    fn add_transformation(&mut self, definition: TransformerDefinition) -> PyResult<()> {
        if !self.core.stages().is_empty() {
            return Err(PyValueError::new_err(
                "cannot append base transformations after stages; use scalar_stage(ScalarStage(...))",
            ));
        }
        let mut definitions = self.core.transformation_definitions().to_vec();
        definitions.push(definition);
        let capacity = if self.explicit_capacity {
            self.core.feature_vector_capacity()
        } else {
            definitions.len()
        };
        let candidate = CorePipelineSpec::with_metadata(
            self.core.raw_feature_extractor_spec().clone(),
            definitions,
            capacity,
            self.core.checksum().map(str::to_owned),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.core = candidate;
        Ok(())
    }

    fn add_stage(&mut self, stage: FittedStage) -> PyResult<()> {
        let capacity = if self.explicit_capacity {
            self.core.feature_vector_capacity()
        } else {
            stage.outputs().len()
        };
        let stages = self.core.stages().iter().cloned().chain([stage]);
        let candidate = CorePipelineSpec::with_stages(
            self.core.raw_feature_extractor_spec().clone(),
            self.core.transformation_definitions().to_vec(),
            stages,
            capacity,
            self.core.checksum().map(str::to_owned),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.core = candidate;
        Ok(())
    }
}

#[pymethods]
impl PipelineSpec {
    /// Allow temporary training prefixes to exceed the configured final capacity.
    fn _fit_candidate(&self) -> Self {
        Self {
            explicit_capacity: false,
            ..self.clone()
        }
    }

    /// Restore the authored final capacity after all intermediate stages have been fitted.
    fn _finalize_fit(&mut self, base: PyRef<'_, Self>) -> PyResult<()> {
        if base.explicit_capacity {
            self.core = CorePipelineSpec::with_stages(
                self.core.raw_feature_extractor_spec().clone(),
                self.core.transformation_definitions().to_vec(),
                self.core.stages().to_vec(),
                base.core.feature_vector_capacity(),
                self.core.checksum().map(str::to_owned),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        self.explicit_capacity = base.explicit_capacity;
        Ok(())
    }

    /// Append independent scalar outputs consuming the current active layout.
    fn scalar_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        stage: PyRef<'_, ScalarStage>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_stage(FittedStage::Scalar {
            transformations: stage.definitions.clone(),
        })?;
        Ok(slf)
    }

    /// Number of already fitted vector stages.
    #[getter]
    fn stage_count(&self) -> usize {
        self.core.stages().len()
    }

    /// Independent snapshot retaining whether final capacity was explicitly authored.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Append a fitted vector scaler, preserving the current active IDs.
    fn scale_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let outputs = slf.core.output_ids();
        slf.add_stage(FittedStage::StandardScale {
            outputs,
            mean,
            scale,
        })?;
        Ok(slf)
    }

    /// Append fitted sklearn MinMaxScaler state, preserving the current active IDs.
    #[pyo3(signature = (scale, min, clip=None))]
    fn min_max_scale_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        scale: Vec<f64>,
        min: Vec<f64>,
        clip: Option<(f64, f64)>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let outputs = slf.core.output_ids();
        slf.add_stage(FittedStage::MinMaxScale {
            outputs,
            scale,
            min,
            clip,
        })?;
        Ok(slf)
    }

    /// Append fitted sklearn PowerTransformer state, preserving the current active IDs.
    fn power_transform_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        method: String,
        lambdas: Vec<f64>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let outputs = slf.core.output_ids();
        slf.add_stage(FittedStage::PowerTransform {
            outputs,
            method,
            lambdas,
            mean,
            scale,
        })?;
        Ok(slf)
    }

    /// Append an ordered fitted column selection, preserving retained input IDs.
    fn select_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        input_indices: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let inputs = slf.core.output_ids();
        let outputs = input_indices
            .iter()
            .map(|&index| {
                inputs.get(index).cloned().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "selection index {index} is out of bounds for {} inputs",
                        inputs.len()
                    ))
                })
            })
            .collect::<PyResult<_>>()?;
        slf.add_stage(FittedStage::Select {
            outputs,
            input_indices,
        })?;
        Ok(slf)
    }

    /// Append fitted sklearn SimpleImputer state and its frozen output layout.
    fn simple_impute_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        outputs: Vec<String>,
        retained_input_indices: Vec<usize>,
        replacement_values: Vec<f64>,
        indicator_input_indices: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_stage(FittedStage::SimpleImpute {
            outputs: outputs.into_iter().map(FeatureId::new).collect(),
            retained_input_indices,
            replacement_values,
            indicator_input_indices,
        })?;
        Ok(slf)
    }

    /// Append fitted PCA state; component rows follow output order.
    fn pca_stage<'py>(
        mut slf: PyRefMut<'py, Self>,
        outputs: Vec<String>,
        mean: Vec<f64>,
        components: Vec<Vec<f64>>,
        output_scale: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_stage(FittedStage::Pca {
            outputs: outputs.into_iter().map(FeatureId::new).collect(),
            mean,
            components,
            output_scale,
        })?;
        Ok(slf)
    }

    #[new]
    #[pyo3(signature = (raw_feature_extractor_spec, *, capacity=None, checksum=None))]
    fn new(
        raw_feature_extractor_spec: PyRef<'_, FeatureExtractorSpec>,
        capacity: Option<usize>,
        checksum: Option<String>,
    ) -> PyResult<Self> {
        let explicit_capacity = capacity.is_some();
        let core = CorePipelineSpec::with_metadata(
            raw_feature_extractor_spec.core.clone(),
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
        slf.add_transformation(TransformerDefinition::identity(
            FeatureId::new(input),
            FeatureId::new(output),
        ))?;
        Ok(slf)
    }

    /// Append a raw scalar from `lag_window` accepted events earlier.
    /// The window must be in `1..=10_000`; output remains NaN until enough history exists.
    #[pyo3(signature = (input, *, lag_window, output=None))]
    fn lagged<'py>(
        mut slf: PyRefMut<'py, Self>,
        input: &str,
        lag_window: usize,
        output: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let output = output.unwrap_or(input);
        slf.add_transformation(TransformerDefinition::lagged(
            FeatureId::new(input),
            FeatureId::new(output),
            lag_window,
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
        slf.add_transformation(TransformerDefinition::standard_scale(
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
        core_feature_ids(self.core.raw_feature_extractor_spec())
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

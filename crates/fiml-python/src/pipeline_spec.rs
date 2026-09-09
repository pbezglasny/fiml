//! Defines model transformations and the Python-facing pipeline specification.

use fiml::{
    FeatureExtractorSpec as CoreFeatureExtractorSpec, FeatureId, PipelineSpec as CorePipelineSpec,
    TransformerDefinition,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::feature_extractor_spec::FeatureExtractorSpec;

pub(crate) fn core_feature_ids(spec: &CoreFeatureExtractorSpec) -> Vec<String> {
    spec.definitions()
        .iter()
        .map(|definition| definition.id.as_str().to_owned())
        .collect()
}

pub(crate) fn model_output_ids(spec: &CorePipelineSpec) -> Vec<String> {
    spec.transformation_definitions()
        .iter()
        .map(|definition| match definition {
            TransformerDefinition::Identity { output, .. }
            | TransformerDefinition::Lagged { output, .. }
            | TransformerDefinition::StandardScale { output, .. } => output.as_str().to_owned(),
        })
        .collect()
}

/// Validated raw-feature and fitted scalar-transformation specification.
///
/// The raw spec is cloned at construction so later Python builder mutations do
/// not change the model artifact. Transformations remain in authored order.
#[pyclass]
pub struct PipelineSpec {
    pub(crate) core: CorePipelineSpec,
    explicit_capacity: bool,
}

impl PipelineSpec {
    fn add_transformation(&mut self, definition: TransformerDefinition) -> PyResult<()> {
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
}

#[pymethods]
impl PipelineSpec {
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

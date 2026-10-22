//! Serializes and deserializes versioned model-input pipeline specifications.
//!
use ::serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{FeatureExtractorSpec, serialization::deserialize_present_option};
use crate::{FeatureId, FittedStage, PipelineSpec, TransformerDefinition, TransformerOperation};

const FORMAT_VERSION: &str = "4.0";

/// Private versioned storage representation for a complete model-input layout.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineSpecWire {
    version: String,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    checksum: Option<String>,
    nodes: Vec<NodeWire>,
    outputs: Vec<OutputWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum NodeWire {
    Extractor { spec: FeatureExtractorSpec },
    Transform { parent: usize, stage: StageWire },
}

/// Ordered named reference into the artifact-local node array.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputWire {
    name: String,
    node: usize,
}

/// Strict numeric storage for fitted stages, independent of compiled runtime state.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum StageWire {
    RollingPathSignature {
        input: String,
        outputs: Vec<String>,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    Scalar {
        transformations: Vec<TransformationWire>,
    },
    Select {
        outputs: Vec<String>,
        input_indices: Vec<usize>,
    },
    StandardScale {
        outputs: Vec<String>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    },
    MinMaxScale {
        outputs: Vec<String>,
        scale: Vec<f64>,
        min: Vec<f64>,
        #[serde(
            default,
            deserialize_with = "deserialize_present_option",
            skip_serializing_if = "Option::is_none"
        )]
        clip: Option<(f64, f64)>,
    },
    PowerTransform {
        outputs: Vec<String>,
        method: String,
        lambdas: Vec<f64>,
        mean: Vec<f64>,
        scale: Vec<f64>,
    },
    QuantileTransform {
        outputs: Vec<String>,
        output_distribution: String,
        quantiles: Vec<Vec<f64>>,
        references: Vec<f64>,
        all_nan_input_indices: Vec<usize>,
        bounds_threshold: f64,
        #[serde(
            default,
            deserialize_with = "deserialize_present_option",
            skip_serializing_if = "Option::is_none"
        )]
        normal_clip: Option<(f64, f64)>,
    },
    SimpleImpute {
        outputs: Vec<String>,
        retained_input_indices: Vec<usize>,
        replacement_values: Vec<f64>,
        indicator_input_indices: Vec<usize>,
    },
    Pca {
        outputs: Vec<String>,
        mean: Vec<f64>,
        components: Vec<Vec<f64>>,
        output_scale: Vec<f64>,
    },
}

impl From<&FittedStage> for StageWire {
    fn from(stage: &FittedStage) -> Self {
        let outputs = stage
            .outputs()
            .iter()
            .map(|id| id.as_str().to_owned())
            .collect();
        match stage {
            FittedStage::RollingPathSignature {
                input,
                window,
                warmup_policy,
                ..
            } => Self::RollingPathSignature {
                input: input.as_str().to_owned(),
                outputs,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            FittedStage::Scalar { transformations } => Self::Scalar {
                transformations: transformations
                    .iter()
                    .map(TransformationWire::from)
                    .collect(),
            },
            FittedStage::Select { input_indices, .. } => Self::Select {
                outputs,
                input_indices: input_indices.clone(),
            },
            FittedStage::StandardScale { mean, scale, .. } => Self::StandardScale {
                outputs,
                mean: mean.clone(),
                scale: scale.clone(),
            },
            FittedStage::MinMaxScale {
                scale, min, clip, ..
            } => Self::MinMaxScale {
                outputs,
                scale: scale.clone(),
                min: min.clone(),
                clip: *clip,
            },
            FittedStage::PowerTransform {
                method,
                lambdas,
                mean,
                scale,
                ..
            } => Self::PowerTransform {
                outputs,
                method: method.clone(),
                lambdas: lambdas.clone(),
                mean: mean.clone(),
                scale: scale.clone(),
            },
            FittedStage::QuantileTransform {
                output_distribution,
                quantiles,
                references,
                all_nan_input_indices,
                bounds_threshold,
                normal_clip,
                ..
            } => Self::QuantileTransform {
                outputs,
                output_distribution: output_distribution.clone(),
                quantiles: quantiles.clone(),
                references: references.clone(),
                all_nan_input_indices: all_nan_input_indices.clone(),
                bounds_threshold: *bounds_threshold,
                normal_clip: *normal_clip,
            },
            FittedStage::SimpleImpute {
                retained_input_indices,
                replacement_values,
                indicator_input_indices,
                ..
            } => Self::SimpleImpute {
                outputs,
                retained_input_indices: retained_input_indices.clone(),
                replacement_values: replacement_values.clone(),
                indicator_input_indices: indicator_input_indices.clone(),
            },
            FittedStage::Pca {
                mean,
                components,
                output_scale,
                ..
            } => Self::Pca {
                outputs,
                mean: mean.clone(),
                components: components.clone(),
                output_scale: output_scale.clone(),
            },
        }
    }
}

impl From<StageWire> for FittedStage {
    fn from(stage: StageWire) -> Self {
        match stage {
            StageWire::RollingPathSignature {
                input,
                outputs,
                window,
                warmup_policy,
            } => Self::RollingPathSignature {
                input: FeatureId::new(input),
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                window,
                warmup_policy,
            },
            StageWire::Scalar { transformations } => Self::Scalar {
                transformations: transformations
                    .into_iter()
                    .map(TransformerDefinition::from)
                    .collect(),
            },
            StageWire::Select {
                outputs,
                input_indices,
            } => Self::Select {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                input_indices,
            },
            StageWire::StandardScale {
                outputs,
                mean,
                scale,
            } => Self::StandardScale {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                mean,
                scale,
            },
            StageWire::MinMaxScale {
                outputs,
                scale,
                min,
                clip,
            } => Self::MinMaxScale {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                scale,
                min,
                clip,
            },
            StageWire::PowerTransform {
                outputs,
                method,
                lambdas,
                mean,
                scale,
            } => Self::PowerTransform {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                method,
                lambdas,
                mean,
                scale,
            },
            StageWire::QuantileTransform {
                outputs,
                output_distribution,
                quantiles,
                references,
                all_nan_input_indices,
                bounds_threshold,
                normal_clip,
            } => Self::QuantileTransform {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                output_distribution,
                quantiles,
                references,
                all_nan_input_indices,
                bounds_threshold,
                normal_clip,
            },
            StageWire::SimpleImpute {
                outputs,
                retained_input_indices,
                replacement_values,
                indicator_input_indices,
            } => Self::SimpleImpute {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                retained_input_indices,
                replacement_values,
                indicator_input_indices,
            },
            StageWire::Pca {
                outputs,
                mean,
                components,
                output_scale,
            } => Self::Pca {
                outputs: outputs.into_iter().map(FeatureId::new).collect(),
                mean,
                components,
                output_scale,
            },
        }
    }
}

/// Private ID-based representation of the supported scalar transformations.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum TransformationWire {
    PositiveStreak {
        input: String,
        output: String,
        epsilon: f64,
    },
    NegativeStreak {
        input: String,
        output: String,
        epsilon: f64,
    },
    ObservationsSincePositive {
        input: String,
        output: String,
        epsilon: f64,
    },
    ObservationsSinceNegative {
        input: String,
        output: String,
        epsilon: f64,
    },
    ObservationsSinceNonzero {
        input: String,
        output: String,
        epsilon: f64,
    },
    TimeSincePositive {
        input: String,
        output: String,
        epsilon: f64,
    },
    TimeSinceNegative {
        input: String,
        output: String,
        epsilon: f64,
    },
    TimeSinceNonzero {
        input: String,
        output: String,
        epsilon: f64,
    },
    RollingSlope {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingPathLength {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingPositiveMovement {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingNegativeMovement {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingDirectionChangeCount {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingMaxDrawdown {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingStd {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingStdTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingMin {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingMinTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingMax {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingMaxTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingRange {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    RollingRangeTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
        warmup_policy: crate::WarmupPolicy,
    },

    LaggedTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        lag_window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
    },
    DeltaTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        lag_window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
    },
    SimpleReturnTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        lag_window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
    },
    LogReturnTimed {
        input: String,
        output: String,
        #[serde(with = "lag_duration")]
        lag_window: std::time::Duration,
        #[serde(
            default = "lag_duration::default_aggregation",
            with = "lag_duration",
            skip_serializing_if = "lag_duration::is_default_aggregation"
        )]
        aggregation: std::time::Duration,
    },

    Delta {
        input: String,
        output: String,
        lag_window: usize,
    },
    SimpleReturn {
        input: String,
        output: String,
        lag_window: usize,
    },
    LogReturn {
        input: String,
        output: String,
        lag_window: usize,
    },

    Ema {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    Sma {
        input: String,
        output: String,
        window: usize,
        warmup_policy: crate::WarmupPolicy,
    },
    Identity {
        input: String,
        output: String,
    },
    Lagged {
        input: String,
        output: String,
        lag_window: usize,
    },
    StandardScale {
        input: String,
        output: String,
        mean: f64,
        scale: f64,
    },
}

impl Serialize for FittedStage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        StageWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FittedStage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        StageWire::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for PipelineSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(::serde::ser::Error::custom)?;
        PipelineSpecWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PipelineSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PipelineSpecWire::deserialize(deserializer)?;
        PipelineSpec::try_from(wire).map_err(::serde::de::Error::custom)
    }
}

impl From<&PipelineSpec> for PipelineSpecWire {
    fn from(spec: &PipelineSpec) -> Self {
        Self {
            version: FORMAT_VERSION.to_owned(),
            checksum: spec.checksum().map(str::to_owned),
            nodes: std::iter::once(NodeWire::Extractor {
                spec: spec.raw_feature_extractor_spec().clone(),
            })
            .chain(spec.nodes().iter().map(|node| NodeWire::Transform {
                parent: node.parent,
                stage: StageWire::from(&node.stage),
            }))
            .collect(),
            outputs: spec
                .outputs()
                .iter()
                .map(|output| OutputWire {
                    name: output.name.clone(),
                    node: output.node,
                })
                .collect(),
        }
    }
}

impl From<&TransformerDefinition> for TransformationWire {
    fn from(definition: &TransformerDefinition) -> Self {
        let input = definition.input.as_str().to_owned();
        let output = definition.output.as_str().to_owned();
        match &definition.operation {
            TransformerOperation::PositiveStreak { epsilon } => Self::PositiveStreak {
                input,
                output,
                epsilon: *epsilon,
            },
            TransformerOperation::NegativeStreak { epsilon } => Self::NegativeStreak {
                input,
                output,
                epsilon: *epsilon,
            },
            TransformerOperation::ObservationsSincePositive { epsilon } => {
                Self::ObservationsSincePositive {
                    input,
                    output,
                    epsilon: *epsilon,
                }
            }
            TransformerOperation::ObservationsSinceNegative { epsilon } => {
                Self::ObservationsSinceNegative {
                    input,
                    output,
                    epsilon: *epsilon,
                }
            }
            TransformerOperation::ObservationsSinceNonzero { epsilon } => {
                Self::ObservationsSinceNonzero {
                    input,
                    output,
                    epsilon: *epsilon,
                }
            }
            TransformerOperation::TimeSincePositive { epsilon } => Self::TimeSincePositive {
                input,
                output,
                epsilon: *epsilon,
            },
            TransformerOperation::TimeSinceNegative { epsilon } => Self::TimeSinceNegative {
                input,
                output,
                epsilon: *epsilon,
            },
            TransformerOperation::TimeSinceNonzero { epsilon } => Self::TimeSinceNonzero {
                input,
                output,
                epsilon: *epsilon,
            },
            TransformerOperation::RollingSlope {
                window,
                warmup_policy,
            } => Self::RollingSlope {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingPathLength {
                window,
                warmup_policy,
            } => Self::RollingPathLength {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingPositiveMovement {
                window,
                warmup_policy,
            } => Self::RollingPositiveMovement {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingNegativeMovement {
                window,
                warmup_policy,
            } => Self::RollingNegativeMovement {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingDirectionChangeCount {
                window,
                warmup_policy,
            } => Self::RollingDirectionChangeCount {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingMaxDrawdown {
                window,
                warmup_policy,
            } => Self::RollingMaxDrawdown {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingStd {
                window,
                warmup_policy,
            } => Self::RollingStd {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingStdTimed {
                window,
                aggregation,
                warmup_policy,
            } => Self::RollingStdTimed {
                input,
                output,
                window: *window,
                aggregation: *aggregation,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingMin {
                window,
                warmup_policy,
            } => Self::RollingMin {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingMinTimed {
                window,
                aggregation,
                warmup_policy,
            } => Self::RollingMinTimed {
                input,
                output,
                window: *window,
                aggregation: *aggregation,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingMax {
                window,
                warmup_policy,
            } => Self::RollingMax {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingMaxTimed {
                window,
                aggregation,
                warmup_policy,
            } => Self::RollingMaxTimed {
                input,
                output,
                window: *window,
                aggregation: *aggregation,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingRange {
                window,
                warmup_policy,
            } => Self::RollingRange {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::RollingRangeTimed {
                window,
                aggregation,
                warmup_policy,
            } => Self::RollingRangeTimed {
                input,
                output,
                window: *window,
                aggregation: *aggregation,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::LaggedTimed {
                lag_window,
                aggregation,
            } => Self::LaggedTimed {
                input,
                output,
                lag_window: *lag_window,
                aggregation: *aggregation,
            },
            TransformerOperation::DeltaTimed {
                lag_window,
                aggregation,
            } => Self::DeltaTimed {
                input,
                output,
                lag_window: *lag_window,
                aggregation: *aggregation,
            },
            TransformerOperation::SimpleReturnTimed {
                lag_window,
                aggregation,
            } => Self::SimpleReturnTimed {
                input,
                output,
                lag_window: *lag_window,
                aggregation: *aggregation,
            },
            TransformerOperation::LogReturnTimed {
                lag_window,
                aggregation,
            } => Self::LogReturnTimed {
                input,
                output,
                lag_window: *lag_window,
                aggregation: *aggregation,
            },

            TransformerOperation::Ema {
                window,
                warmup_policy,
            } => Self::Ema {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::Sma {
                window,
                warmup_policy,
            } => Self::Sma {
                input,
                output,
                window: *window,
                warmup_policy: *warmup_policy,
            },
            TransformerOperation::Delta { lag_window } => Self::Delta {
                input,
                output,
                lag_window: *lag_window,
            },
            TransformerOperation::SimpleReturn { lag_window } => Self::SimpleReturn {
                input,
                output,
                lag_window: *lag_window,
            },
            TransformerOperation::LogReturn { lag_window } => Self::LogReturn {
                input,
                output,
                lag_window: *lag_window,
            },
            TransformerOperation::Identity => Self::Identity { input, output },
            TransformerOperation::Lagged { lag_window } => Self::Lagged {
                input,
                output,
                lag_window: *lag_window,
            },
            TransformerOperation::StandardScale { mean, scale } => Self::StandardScale {
                input,
                output,
                mean: *mean,
                scale: *scale,
            },
        }
    }
}

impl TryFrom<PipelineSpecWire> for PipelineSpec {
    type Error = String;

    fn try_from(wire: PipelineSpecWire) -> Result<Self, Self::Error> {
        if wire.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported pipeline spec version {:?}; expected {FORMAT_VERSION}",
                wire.version
            ));
        }
        let mut nodes = wire.nodes.into_iter();
        let Some(NodeWire::Extractor { spec: extractor }) = nodes.next() else {
            return Err("node 0 must be the extractor".into());
        };
        let mut spec = PipelineSpec::new(extractor);
        spec.set_checksum(wire.checksum);
        for (offset, node) in nodes.enumerate() {
            let index = offset + 1;
            let NodeWire::Transform { parent, stage } = node else {
                return Err(format!("node {index}: only node 0 may be an extractor"));
            };
            let parent = spec
                .node(parent)
                .ok_or_else(|| format!("node {index}: parent must refer to an earlier node"))?;
            spec.add(parent, stage.into())
                .map_err(|error| error.to_string())?;
        }
        for output in wire.outputs {
            let node = spec
                .node(output.node)
                .ok_or_else(|| format!("output {:?}: node index out of bounds", output.name))?;
            spec.output(output.name, node)
                .map_err(|error| error.to_string())?;
        }
        spec.validate().map_err(|error| error.to_string())?;
        Ok(spec)
    }
}

impl From<TransformationWire> for TransformerDefinition {
    fn from(transformation: TransformationWire) -> Self {
        match transformation {
            TransformationWire::PositiveStreak {
                input,
                output,
                epsilon,
            } => Self::positive_streak(FeatureId::new(input), FeatureId::new(output), epsilon),
            TransformationWire::NegativeStreak {
                input,
                output,
                epsilon,
            } => Self::negative_streak(FeatureId::new(input), FeatureId::new(output), epsilon),
            TransformationWire::ObservationsSincePositive {
                input,
                output,
                epsilon,
            } => Self::observations_since_positive(
                FeatureId::new(input),
                FeatureId::new(output),
                epsilon,
            ),
            TransformationWire::ObservationsSinceNegative {
                input,
                output,
                epsilon,
            } => Self::observations_since_negative(
                FeatureId::new(input),
                FeatureId::new(output),
                epsilon,
            ),
            TransformationWire::ObservationsSinceNonzero {
                input,
                output,
                epsilon,
            } => Self::observations_since_nonzero(
                FeatureId::new(input),
                FeatureId::new(output),
                epsilon,
            ),
            TransformationWire::TimeSincePositive {
                input,
                output,
                epsilon,
            } => Self::time_since_positive(FeatureId::new(input), FeatureId::new(output), epsilon),
            TransformationWire::TimeSinceNegative {
                input,
                output,
                epsilon,
            } => Self::time_since_negative(FeatureId::new(input), FeatureId::new(output), epsilon),
            TransformationWire::TimeSinceNonzero {
                input,
                output,
                epsilon,
            } => Self::time_since_nonzero(FeatureId::new(input), FeatureId::new(output), epsilon),
            TransformationWire::RollingSlope {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_slope(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingPathLength {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_path_length(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingPositiveMovement {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_positive_movement(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingNegativeMovement {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_negative_movement(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingDirectionChangeCount {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_direction_change_count(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingMaxDrawdown {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_max_drawdown(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingStd {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_std(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingStdTimed {
                input,
                output,
                window,
                aggregation,
                warmup_policy,
            } => Self::rolling_std_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                aggregation,
                warmup_policy,
            ),
            TransformationWire::RollingMin {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_min(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingMinTimed {
                input,
                output,
                window,
                aggregation,
                warmup_policy,
            } => Self::rolling_min_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                aggregation,
                warmup_policy,
            ),
            TransformationWire::RollingMax {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_max(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingMaxTimed {
                input,
                output,
                window,
                aggregation,
                warmup_policy,
            } => Self::rolling_max_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                aggregation,
                warmup_policy,
            ),
            TransformationWire::RollingRange {
                input,
                output,
                window,
                warmup_policy,
            } => Self::rolling_range(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::RollingRangeTimed {
                input,
                output,
                window,
                aggregation,
                warmup_policy,
            } => Self::rolling_range_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                aggregation,
                warmup_policy,
            ),
            TransformationWire::LaggedTimed {
                input,
                output,
                lag_window,
                aggregation,
            } => Self::lagged_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                lag_window,
                aggregation,
            ),
            TransformationWire::DeltaTimed {
                input,
                output,
                lag_window,
                aggregation,
            } => Self::delta_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                lag_window,
                aggregation,
            ),
            TransformationWire::SimpleReturnTimed {
                input,
                output,
                lag_window,
                aggregation,
            } => Self::simple_return_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                lag_window,
                aggregation,
            ),
            TransformationWire::LogReturnTimed {
                input,
                output,
                lag_window,
                aggregation,
            } => Self::log_return_timed_with_aggregation(
                FeatureId::new(input),
                FeatureId::new(output),
                lag_window,
                aggregation,
            ),

            TransformationWire::Ema {
                input,
                output,
                window,
                warmup_policy,
            } => Self::ema(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::Sma {
                input,
                output,
                window,
                warmup_policy,
            } => Self::sma(
                FeatureId::new(input),
                FeatureId::new(output),
                window,
                warmup_policy,
            ),
            TransformationWire::Delta {
                input,
                output,
                lag_window,
            } => Self::delta(FeatureId::new(input), FeatureId::new(output), lag_window),
            TransformationWire::SimpleReturn {
                input,
                output,
                lag_window,
            } => Self::simple_return(FeatureId::new(input), FeatureId::new(output), lag_window),
            TransformationWire::LogReturn {
                input,
                output,
                lag_window,
            } => Self::log_return(FeatureId::new(input), FeatureId::new(output), lag_window),
            TransformationWire::Identity { input, output } => {
                Self::identity(FeatureId::new(input), FeatureId::new(output))
            }
            TransformationWire::Lagged {
                input,
                output,
                lag_window,
            } => Self::lagged(FeatureId::new(input), FeatureId::new(output), lag_window),
            TransformationWire::StandardScale {
                input,
                output,
                mean,
                scale,
            } => Self::standard_scale(FeatureId::new(input), FeatureId::new(output), mean, scale),
        }
    }
}

/// Reuse the extractor artifact's strict whole-millisecond duration representation.
mod lag_duration {
    use super::*;
    use std::time::Duration;

    pub(super) fn default_aggregation() -> Duration {
        Duration::from_millis(1)
    }

    pub(super) fn is_default_aggregation(duration: &Duration) -> bool {
        *duration == default_aggregation()
    }

    pub(super) fn serialize<S: Serializer>(
        duration: &Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let text = super::super::serialization::format_duration(*duration)
            .map_err(::serde::ser::Error::custom)?;
        serializer.serialize_str(&text)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Duration, D::Error> {
        let text = String::deserialize(deserializer)?;
        super::super::serialization::parse_duration(&text).map_err(::serde::de::Error::custom)
    }
}

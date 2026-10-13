//! Defines, validates, and executes scalar transformations for model input.
//!
//! Compilation resolves each transformation to indexes and precomputed parameters
//! so execution writes directly into caller-owned storage without allocation.

use std::{collections::HashMap, time::Duration};

use crate::{
    FeatureId, FeatureVector, FimlError, InvalidTransformationDefinitionError, Result, WarmupPolicy,
};

mod average;
use average::AverageTransformer;
mod change;
mod identity;
mod lagged;
mod standard_scale;
mod timed;

use timed::TimedTransformer;

use change::{ChangeKind, ChangeTransformer};
use identity::IdentityTransformer;
use lagged::LaggedFeature;
use standard_scale::StandardScaleTransformer;

const WINDOW_MAX_SIZE: usize = 10_000;

/// One named scalar transformation from an input layout to the next output layout.
///
/// Timed variants use the newest finite observation at or before `clock - lag_window`.
/// Single-symbol clocks advance on every accepted event for that symbol; global calendars
/// and event-count lags use the maximum accepted timestamp. Column-wise stages preserve
/// clocks, and PCA combines them using their maximum. Observations behind a composite
/// clock replace the last finite value at that clock. Context-only inputs have no clock.
///
/// Timed outputs stay NaN until an eligible observation exists. Missing observations
/// preserve history but emit NaN until the next finite observation. Clock-only refreshes
/// are observations of the timed output, without adding samples to its input history.
/// All timed kinds/windows of one input share history within a stage, reserving approximately
/// 16 bytes per millisecond of the maximum lag. Same-millisecond finite observations replace
/// one another. Context updates do not advance timed histories or refresh their outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerDefinition {
    /// Emits the as-of value at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    LaggedTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration; history costs about 16 bytes per millisecond.
        lag_window: Duration,
    },
    /// Emits current minus the as-of value at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    DeltaTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration; history costs about 16 bytes per millisecond.
        lag_window: Duration,
    },
    /// Emits current / as-of value - 1 at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    SimpleReturnTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration; history costs about 16 bytes per millisecond.
        lag_window: Duration,
    },
    /// Emits ln(current) - ln(as-of value) at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    LogReturnTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration; history costs about 16 bytes per millisecond.
        lag_window: Duration,
    },

    /// EMA over finite source observations; missing observations emit NaN without advancing history.
    Ema {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// SMA over finite source observations; missing observations emit NaN without advancing history.
    Sma {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Computes `current - previous` over finite source observations.
    /// A lag of `k` requires `k + 1` observations; until then the output is NaN.
    /// Observed nonfinite values emit NaN without advancing history; unrelated events retain output.
    /// All change kinds and windows for the same input share history within a stage.
    Delta {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Number of finite observations to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Computes `current / previous - 1` over finite source observations.
    /// A zero denominator emits NaN.
    /// A lag of `k` requires `k + 1` observations; until then the output is NaN.
    /// Observed nonfinite values emit NaN without advancing history; unrelated events retain output.
    /// All change kinds and windows for the same input share history within a stage.
    SimpleReturn {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Number of finite observations to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Computes `ln(current) - ln(previous)` over finite source observations.
    /// Nonpositive operands emit NaN.
    /// A lag of `k` requires `k + 1` observations; until then the output is NaN.
    /// Observed nonfinite values emit NaN without advancing history; unrelated events retain output.
    /// All change kinds and windows for the same input share history within a stage.
    LogReturn {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Number of finite observations to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Copies one input scalar without changing its value.
    Identity {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
    },
    /// Emits the input scalar from `lag_window` accepted events earlier.
    /// The window must be in `1..=10_000`; output remains NaN until enough history exists.
    /// Definitions for the same input share one runtime history buffer.
    Lagged {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Number of accepted events to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Applies `(input - mean) / scale` to one input scalar.
    StandardScale {
        /// ID to read from the preceding stage's layout.
        input: FeatureId,
        /// Unique, non-reserved ID assigned to this output.
        output: FeatureId,
        /// Finite centering value.
        mean: f64,
        /// Positive finite divisor whose reciprocal must also be finite.
        scale: f64,
    },
}

impl TransformerDefinition {
    /// Creates a source-clock transformation emitting the as-of value.
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn lagged_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::LaggedTimed {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a source-clock transformation emitting current minus the as-of value.
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn delta_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::DeltaTimed {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a source-clock transformation emitting current / as-of value - 1.
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn simple_return_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::SimpleReturnTimed {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a source-clock transformation emitting ln(current) - ln(as-of value).
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn log_return_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::LogReturnTimed {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a sample-based EMA transformation.
    pub fn ema(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::Ema {
            input,
            output,
            window,
            warmup_policy,
        }
    }

    /// Creates a sample-based SMA transformation.
    pub fn sma(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::Sma {
            input,
            output,
            window,
            warmup_policy,
        }
    }

    /// Creates an exact observation-based delta transformation.
    pub fn delta(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self::Delta {
            input,
            output,
            lag_window,
        }
    }

    /// Creates an exact observation-based simple return transformation.
    pub fn simple_return(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self::SimpleReturn {
            input,
            output,
            lag_window,
        }
    }

    /// Creates an exact observation-based log return transformation.
    pub fn log_return(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self::LogReturn {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a scalar identity transformation.
    pub fn identity(input: FeatureId, output: FeatureId) -> Self {
        Self::Identity { input, output }
    }

    /// Creates an event-lagged transformation with a positive history window.
    pub fn lagged(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self::Lagged {
            input,
            output,
            lag_window,
        }
    }

    /// Creates a scalar standard-scaling transformation.
    pub fn standard_scale(input: FeatureId, output: FeatureId, mean: f64, scale: f64) -> Self {
        Self::StandardScale {
            input,
            output,
            mean,
            scale,
        }
    }

    pub(crate) fn input(&self) -> &FeatureId {
        match self {
            Self::LaggedTimed { input, .. }
            | Self::DeltaTimed { input, .. }
            | Self::SimpleReturnTimed { input, .. }
            | Self::LogReturnTimed { input, .. }
            | Self::Sma { input, .. }
            | Self::Ema { input, .. }
            | Self::Identity { input, .. }
            | Self::Delta { input, .. }
            | Self::SimpleReturn { input, .. }
            | Self::LogReturn { input, .. }
            | Self::Lagged { input, .. }
            | Self::StandardScale { input, .. } => input,
        }
    }

    pub(crate) fn output(&self) -> &FeatureId {
        match self {
            Self::LaggedTimed { output, .. }
            | Self::DeltaTimed { output, .. }
            | Self::SimpleReturnTimed { output, .. }
            | Self::LogReturnTimed { output, .. }
            | Self::Sma { output, .. }
            | Self::Ema { output, .. }
            | Self::Identity { output, .. }
            | Self::Delta { output, .. }
            | Self::SimpleReturn { output, .. }
            | Self::LogReturn { output, .. }
            | Self::Lagged { output, .. }
            | Self::StandardScale { output, .. } => output,
        }
    }

    pub(crate) fn timed_window(&self) -> Option<Duration> {
        match self {
            Self::LaggedTimed { lag_window, .. }
            | Self::DeltaTimed { lag_window, .. }
            | Self::SimpleReturnTimed { lag_window, .. }
            | Self::LogReturnTimed { lag_window, .. } => Some(*lag_window),
            _ => None,
        }
    }

    pub(crate) fn validate(&self) -> std::result::Result<(), InvalidTransformationDefinitionError> {
        if let Some(window) = self.timed_window() {
            timed::validate_window(window)?;
        }
        if matches!(
            self,
            Self::Sma { window: 0, .. } | Self::Ema { window: 0, .. }
        ) {
            return Err(InvalidTransformationDefinitionError::WindowZero);
        }
        if let Self::Lagged { lag_window, .. }
        | Self::Delta { lag_window, .. }
        | Self::SimpleReturn { lag_window, .. }
        | Self::LogReturn { lag_window, .. } = self
        {
            if *lag_window == 0 {
                return Err(InvalidTransformationDefinitionError::LagWindowZero);
            }
            if *lag_window > WINDOW_MAX_SIZE {
                return Err(InvalidTransformationDefinitionError::LagWindowTooLarge);
            }
        }
        if let Self::StandardScale { mean, scale, .. } = self {
            validate_standard_scale(*mean, *scale)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_standard_scale(
    mean: f64,
    scale: f64,
) -> std::result::Result<(), InvalidTransformationDefinitionError> {
    if !mean.is_finite() {
        return Err(InvalidTransformationDefinitionError::MeanNotFinite);
    }
    if !scale.is_finite() {
        return Err(InvalidTransformationDefinitionError::ScaleNotFinite);
    }
    if scale <= 0.0 {
        return Err(InvalidTransformationDefinitionError::ScaleNotPositive);
    }
    if !(1.0 / scale).is_finite() {
        return Err(InvalidTransformationDefinitionError::InverseScaleNotFinite);
    }
    Ok(())
}

/// Validates independent scalar outputs against the preceding active layout.
pub(crate) fn validate(definitions: &[TransformerDefinition], inputs: &[FeatureId]) -> Result<()> {
    for (index, definition) in definitions.iter().enumerate() {
        let reason = if !inputs.contains(definition.input()) {
            Some(InvalidTransformationDefinitionError::InputFeatureNotFound)
        } else if crate::features::is_reserved_feature_id(definition.output()) {
            Some(InvalidTransformationDefinitionError::ReservedOutputFeature)
        } else if definitions[..index]
            .iter()
            .any(|previous| previous.output() == definition.output())
        {
            Some(InvalidTransformationDefinitionError::DuplicateOutputFeature)
        } else {
            definition.validate().err()
        };
        let reason = reason.or_else(|| {
            average_key(definition).and_then(|key| {
                (definitions[..=index]
                    .iter()
                    .filter(|d| average_key(d) == Some(key))
                    .count()
                    > crate::features::MAX_OUTPUTS_PER_INDICATOR)
                    .then_some(InvalidTransformationDefinitionError::TooManyAverageOutputs)
            })
        });
        if let Some(reason) = reason {
            return Err(FimlError::InvalidTransformationDefinition { index, reason });
        }
    }
    Ok(())
}

fn average_key(definition: &TransformerDefinition) -> Option<(bool, &FeatureId, WarmupPolicy)> {
    match definition {
        TransformerDefinition::Sma {
            input,
            warmup_policy,
            ..
        } => Some((false, input, *warmup_policy)),
        TransformerDefinition::Ema {
            input,
            warmup_policy,
            ..
        } => Some((true, input, *warmup_policy)),
        _ => None,
    }
}

/// Compiles validated scalar definitions, sharing history between lags of one input.
pub(crate) fn compile(
    definitions: &[TransformerDefinition],
    inputs: &[FeatureId],
) -> Box<[Transformer]> {
    let mut operations = Vec::with_capacity(definitions.len());
    let mut averages = HashMap::<(bool, usize, WarmupPolicy), (Vec<usize>, Vec<usize>)>::new();
    let mut timed_outputs = HashMap::<usize, Vec<(Option<ChangeKind>, i64, usize)>>::new();
    let mut changes = HashMap::<usize, Vec<(ChangeKind, usize, usize)>>::new();
    let mut lagged_outputs = HashMap::<usize, (Vec<usize>, Vec<usize>)>::new();
    for (output_index, definition) in definitions.iter().enumerate() {
        let input_index = inputs
            .iter()
            .position(|id| id == definition.input())
            .expect("pipeline construction validated every input ID");
        match definition {
            TransformerDefinition::LaggedTimed { lag_window, .. }
            | TransformerDefinition::DeltaTimed { lag_window, .. }
            | TransformerDefinition::SimpleReturnTimed { lag_window, .. }
            | TransformerDefinition::LogReturnTimed { lag_window, .. } => {
                let kind = match definition {
                    TransformerDefinition::DeltaTimed { .. } => Some(ChangeKind::Delta),
                    TransformerDefinition::SimpleReturnTimed { .. } => {
                        Some(ChangeKind::SimpleReturn)
                    }
                    TransformerDefinition::LogReturnTimed { .. } => Some(ChangeKind::LogReturn),
                    _ => None,
                };
                timed_outputs.entry(input_index).or_default().push((
                    kind,
                    lag_window.as_millis() as i64,
                    output_index,
                ));
            }
            TransformerDefinition::Sma { window, .. }
            | TransformerDefinition::Ema { window, .. } => {
                let (ema, _, warmup) = average_key(definition).unwrap();
                let (windows, outputs) = averages.entry((ema, input_index, warmup)).or_default();
                windows.push(*window);
                outputs.push(output_index);
            }
            TransformerDefinition::Delta { lag_window, .. }
            | TransformerDefinition::SimpleReturn { lag_window, .. }
            | TransformerDefinition::LogReturn { lag_window, .. } => {
                let kind = match definition {
                    TransformerDefinition::Delta { .. } => ChangeKind::Delta,
                    TransformerDefinition::SimpleReturn { .. } => ChangeKind::SimpleReturn,
                    _ => ChangeKind::LogReturn,
                };
                changes
                    .entry(input_index)
                    .or_default()
                    .push((kind, *lag_window, output_index));
            }
            TransformerDefinition::Identity { .. } => operations.push(Transformer::Identity(
                IdentityTransformer::new(input_index, output_index),
            )),
            TransformerDefinition::Lagged { lag_window, .. } => {
                let (windows, output_indices) = lagged_outputs.entry(input_index).or_default();
                windows.push(*lag_window);
                output_indices.push(output_index);
            }
            TransformerDefinition::StandardScale { mean, scale, .. } => {
                operations.push(Transformer::StandardScale(StandardScaleTransformer::new(
                    input_index,
                    output_index,
                    *mean,
                    *scale,
                )));
            }
        }
    }
    // Operations read only the preceding layout and own distinct output cells, so groups
    // can run after scalar operations without changing authored output order.
    for (input_index, (windows, output_indices)) in lagged_outputs {
        operations.push(Transformer::Lagged(LaggedFeature::new(
            input_index,
            windows,
            output_indices.into_boxed_slice(),
        )));
    }
    for ((ema, input, warmup), (windows, outputs)) in averages {
        operations.push(Transformer::Average(Box::new(AverageTransformer::new(
            ema,
            input,
            &windows,
            outputs.into_boxed_slice(),
            warmup,
        ))));
    }
    for (input, outputs) in changes {
        operations.push(Transformer::Change(ChangeTransformer::new(input, outputs)));
    }
    for (input, outputs) in timed_outputs {
        operations.push(Transformer::Timed(TimedTransformer::new(input, outputs)));
    }
    operations.into_boxed_slice()
}

/// Resolved scalar operation for allocation-free writes into model input.
pub(crate) enum Transformer {
    Timed(TimedTransformer),
    Average(Box<AverageTransformer>),
    Change(ChangeTransformer),
    Identity(IdentityTransformer),
    Lagged(LaggedFeature),
    StandardScale(StandardScaleTransformer),
}

impl Transformer {
    pub(crate) fn apply<V: FeatureVector>(
        &mut self,
        raw_values: &[f64],
        observed: &[bool],
        model_vector: &mut V,
        output_observed: &mut [bool],
        evaluation: Evaluation<'_>,
    ) {
        match self {
            Self::Timed(transformer) => transformer.apply(
                raw_values,
                observed,
                model_vector,
                output_observed,
                evaluation,
            ),
            Self::Average(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
            Self::Change(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
            Self::Identity(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
            Self::Lagged(transformer) => {
                if evaluation.advance {
                    transformer.apply_observed(raw_values, model_vector, output_observed);
                } else {
                    transformer.refresh(model_vector);
                }
            }
            Self::StandardScale(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
        }
    }
}

/// Event execution clocks are separate from observation masks and source freshness.
#[derive(Clone, Copy)]
pub(crate) struct Evaluation<'a> {
    pub(crate) timestamps: &'a [Option<i64>],
    pub(crate) global_timestamp: Option<i64>,
    pub(crate) advance: bool,
}

/// Resolve scalar clock inheritance; event-count lags consume the global arrival stream.
pub(crate) fn clock_inputs(
    definitions: &[TransformerDefinition],
    inputs: &[FeatureId],
) -> Box<[Option<usize>]> {
    definitions
        .iter()
        .map(|definition| {
            if matches!(definition, TransformerDefinition::Lagged { .. }) {
                None
            } else {
                Some(
                    inputs
                        .iter()
                        .position(|id| id == definition.input())
                        .expect("validated scalar input"),
                )
            }
        })
        .collect()
}

pub(crate) fn inherit_clocks(
    sources: &[Option<usize>],
    evaluation: Evaluation<'_>,
    output: &mut [Option<i64>],
) {
    for (&source, clock) in sources.iter().zip(output) {
        *clock = source.map_or(evaluation.global_timestamp, |index| {
            evaluation.timestamps[index]
        });
    }
}

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
mod statistics;
use statistics::{StatisticKind, StatisticsTransformer};
mod timed;

use timed::TimedTransformer;

use change::{ChangeKind, ChangeTransformer};
use identity::IdentityTransformer;
use lagged::LaggedFeature;
use standard_scale::StandardScaleTransformer;

const WINDOW_MAX_SIZE: usize = 10_000;

/// One named scalar transformation from an input layout to the next output layout.
///
/// Timed lag/change variants use the newest retained finite observation at or before `clock - lag_window`.
/// For these lag/change variants, single-symbol clocks advance on every accepted event for that symbol; global calendars
/// and event-count lags use the maximum accepted timestamp. Column-wise stages preserve
/// clocks, and PCA combines them using their maximum. Observations behind a composite
/// clock replace the last finite value at that clock. Context-only inputs have no clock.
///
/// Timed outputs stay NaN until an eligible observation exists. Observed nonfinite values
/// preserve history but emit NaN until the next finite observation. Clock-only refreshes
/// are observations of the timed output, without adding samples to its input history.
/// Timed kinds/windows with the same input and aggregation share history within a stage.
/// Each epoch-aligned bucket retains its last finite observation with its actual timestamp.
/// Aggregation defaults to 1 ms; coarser aggregation discards intermediate observations,
/// but lookups never use an observation after the requested cutoff. A lag must be a positive
/// multiple of aggregation. History reserves `max_lag / aggregation + 2` slots of 16 bytes,
/// capped at 1,000,000 slots (about 16 MB) per shared history. Context updates do not
/// advance timed histories or refresh their outputs.
///
/// Rolling statistics share a summary tree across kinds, windows, and warmup policies
/// for each input (and aggregation for timed statistics). Unlike timed lag/change
/// histories, timed statistics include every finite observation in a bucket, including
/// observations at the same composite clock. Buckets expire when their epoch-aligned
/// start plus the window is at or before the source clock. Full-window warmup measures
/// elapsed time since the first finite observation; empty windows emit NaN. Sample
/// full-window warmup requires the configured number of finite observations.
/// Population standard deviation and range are zero for a single observation.
/// Nonfinite inputs emit NaN without entering history; nonfinite arithmetic results
/// also emit NaN. Timed statistics follow the same clock and context-refresh rules.
/// Statistics updates and distinct-window queries are logarithmic in history size.
/// The tree reserves `2 * next_power_of_two(slots)` summaries of 48 bytes on 64-bit
/// targets, plus 8 bytes per timed slot; timed slots use `max_window / aggregation + 2`
/// and the same 1,000,000-slot limit. Sample slots equal the largest sample window.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerDefinition {
    /// Rolling population standard deviation over finite source observations.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingStd {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling population standard deviation over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingStdTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of aggregation.
        window: Duration,
        /// Positive whole-millisecond bucket width; every finite observation contributes.
        aggregation: Duration,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling minimum over finite source observations.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingMin {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling minimum over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingMinTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of aggregation.
        window: Duration,
        /// Positive whole-millisecond bucket width; every finite observation contributes.
        aggregation: Duration,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling maximum over finite source observations.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingMax {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling maximum over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingMaxTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of aggregation.
        window: Duration,
        /// Positive whole-millisecond bucket width; every finite observation contributes.
        aggregation: Duration,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling range (maximum minus minimum) over finite source observations.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingRange {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling range (maximum minus minimum) over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingRangeTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of aggregation.
        window: Duration,
        /// Positive whole-millisecond bucket width; every finite observation contributes.
        aggregation: Duration,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },

    /// Emits the as-of value at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    LaggedTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },
    /// Emits current minus the as-of value at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    DeltaTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },
    /// Emits current / as-of value - 1 at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    SimpleReturnTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },
    /// Emits ln(current) - ln(as-of value) at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    LogReturnTimed {
        /// ID in the preceding layout.
        input: FeatureId,
        /// ID assigned to this output.
        output: FeatureId,
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
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
    /// Creates an observation-based rolling population standard deviation.
    pub fn rolling_std(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingStd {
            input,
            output,
            window,
            warmup_policy,
        }
    }
    /// Creates a timed rolling population standard deviation, using 1 ms aggregation.
    pub fn rolling_std_timed(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::rolling_std_timed_with_aggregation(
            input,
            output,
            window,
            Duration::from_millis(1),
            warmup_policy,
        )
    }
    /// Creates a timed rolling population standard deviation; buckets include every finite observation.
    pub fn rolling_std_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        aggregation: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingStdTimed {
            input,
            output,
            window,
            aggregation,
            warmup_policy,
        }
    }
    /// Creates an observation-based rolling minimum.
    pub fn rolling_min(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingMin {
            input,
            output,
            window,
            warmup_policy,
        }
    }
    /// Creates a timed rolling minimum, using 1 ms aggregation.
    pub fn rolling_min_timed(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::rolling_min_timed_with_aggregation(
            input,
            output,
            window,
            Duration::from_millis(1),
            warmup_policy,
        )
    }
    /// Creates a timed rolling minimum; buckets include every finite observation.
    pub fn rolling_min_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        aggregation: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingMinTimed {
            input,
            output,
            window,
            aggregation,
            warmup_policy,
        }
    }
    /// Creates an observation-based rolling maximum.
    pub fn rolling_max(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingMax {
            input,
            output,
            window,
            warmup_policy,
        }
    }
    /// Creates a timed rolling maximum, using 1 ms aggregation.
    pub fn rolling_max_timed(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::rolling_max_timed_with_aggregation(
            input,
            output,
            window,
            Duration::from_millis(1),
            warmup_policy,
        )
    }
    /// Creates a timed rolling maximum; buckets include every finite observation.
    pub fn rolling_max_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        aggregation: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingMaxTimed {
            input,
            output,
            window,
            aggregation,
            warmup_policy,
        }
    }
    /// Creates an observation-based rolling range (maximum minus minimum).
    pub fn rolling_range(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingRange {
            input,
            output,
            window,
            warmup_policy,
        }
    }
    /// Creates a timed rolling range (maximum minus minimum), using 1 ms aggregation.
    pub fn rolling_range_timed(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::rolling_range_timed_with_aggregation(
            input,
            output,
            window,
            Duration::from_millis(1),
            warmup_policy,
        )
    }
    /// Creates a timed rolling range (maximum minus minimum); buckets include every finite observation.
    pub fn rolling_range_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        window: Duration,
        aggregation: Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self::RollingRangeTimed {
            input,
            output,
            window,
            aggregation,
            warmup_policy,
        }
    }
    fn statistic(&self) -> Option<(StatisticKind, u64, Option<Duration>, WarmupPolicy)> {
        match self {
            Self::RollingStd {
                window,
                warmup_policy,
                ..
            } => Some((StatisticKind::Std, *window as u64, None, *warmup_policy)),
            Self::RollingStdTimed {
                window,
                warmup_policy,
                aggregation,
                ..
            } => Some((
                StatisticKind::Std,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            Self::RollingMin {
                window,
                warmup_policy,
                ..
            } => Some((StatisticKind::Min, *window as u64, None, *warmup_policy)),
            Self::RollingMinTimed {
                window,
                warmup_policy,
                aggregation,
                ..
            } => Some((
                StatisticKind::Min,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            Self::RollingMax {
                window,
                warmup_policy,
                ..
            } => Some((StatisticKind::Max, *window as u64, None, *warmup_policy)),
            Self::RollingMaxTimed {
                window,
                warmup_policy,
                aggregation,
                ..
            } => Some((
                StatisticKind::Max,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            Self::RollingRange {
                window,
                warmup_policy,
                ..
            } => Some((StatisticKind::Range, *window as u64, None, *warmup_policy)),
            Self::RollingRangeTimed {
                window,
                warmup_policy,
                aggregation,
                ..
            } => Some((
                StatisticKind::Range,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            _ => None,
        }
    }

    /// Creates a source-clock transformation emitting the as-of value.
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn lagged_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::lagged_timed_with_aggregation(input, output, lag_window, Duration::from_millis(1))
    }

    /// Creates a `lagged_timed` transformation retaining the last observation per aggregation bucket.
    /// Coarser aggregation reduces memory and as-of precision; see [`Self`] for limits.
    pub fn lagged_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        lag_window: Duration,
        aggregation: Duration,
    ) -> Self {
        Self::LaggedTimed {
            input,
            output,
            lag_window,
            aggregation,
        }
    }

    /// Creates a source-clock transformation emitting current minus the as-of value.
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn delta_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::delta_timed_with_aggregation(input, output, lag_window, Duration::from_millis(1))
    }

    /// Creates a `delta_timed` transformation retaining the last observation per aggregation bucket.
    /// Coarser aggregation reduces memory and as-of precision; see [`Self`] for limits.
    pub fn delta_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        lag_window: Duration,
        aggregation: Duration,
    ) -> Self {
        Self::DeltaTimed {
            input,
            output,
            lag_window,
            aggregation,
        }
    }

    /// Creates a source-clock transformation emitting current / as-of value - 1.
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn simple_return_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::simple_return_timed_with_aggregation(
            input,
            output,
            lag_window,
            Duration::from_millis(1),
        )
    }

    /// Creates a `simple_return_timed` transformation retaining the last observation per aggregation bucket.
    /// Coarser aggregation reduces memory and as-of precision; see [`Self`] for limits.
    pub fn simple_return_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        lag_window: Duration,
        aggregation: Duration,
    ) -> Self {
        Self::SimpleReturnTimed {
            input,
            output,
            lag_window,
            aggregation,
        }
    }

    /// Creates a source-clock transformation emitting ln(current) - ln(as-of value).
    /// Uses the newest finite observation at or before `clock - lag_window`, without interpolation.
    pub fn log_return_timed(input: FeatureId, output: FeatureId, lag_window: Duration) -> Self {
        Self::log_return_timed_with_aggregation(input, output, lag_window, Duration::from_millis(1))
    }

    /// Creates a `log_return_timed` transformation retaining the last observation per aggregation bucket.
    /// Coarser aggregation reduces memory and as-of precision; see [`Self`] for limits.
    pub fn log_return_timed_with_aggregation(
        input: FeatureId,
        output: FeatureId,
        lag_window: Duration,
        aggregation: Duration,
    ) -> Self {
        Self::LogReturnTimed {
            input,
            output,
            lag_window,
            aggregation,
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
            Self::RollingStd { input, .. }
            | Self::RollingStdTimed { input, .. }
            | Self::RollingMin { input, .. }
            | Self::RollingMinTimed { input, .. }
            | Self::RollingMax { input, .. }
            | Self::RollingMaxTimed { input, .. }
            | Self::RollingRange { input, .. }
            | Self::RollingRangeTimed { input, .. }
            | Self::LaggedTimed { input, .. }
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
            Self::RollingStd { output, .. }
            | Self::RollingStdTimed { output, .. }
            | Self::RollingMin { output, .. }
            | Self::RollingMinTimed { output, .. }
            | Self::RollingMax { output, .. }
            | Self::RollingMaxTimed { output, .. }
            | Self::RollingRange { output, .. }
            | Self::RollingRangeTimed { output, .. }
            | Self::LaggedTimed { output, .. }
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

    pub(crate) fn timed_window(&self) -> Option<(Duration, Duration)> {
        match self {
            Self::RollingStdTimed {
                window,
                aggregation,
                ..
            }
            | Self::RollingMinTimed {
                window,
                aggregation,
                ..
            }
            | Self::RollingMaxTimed {
                window,
                aggregation,
                ..
            }
            | Self::RollingRangeTimed {
                window,
                aggregation,
                ..
            } => Some((*window, *aggregation)),
            Self::LaggedTimed {
                lag_window,
                aggregation,
                ..
            }
            | Self::DeltaTimed {
                lag_window,
                aggregation,
                ..
            }
            | Self::SimpleReturnTimed {
                lag_window,
                aggregation,
                ..
            }
            | Self::LogReturnTimed {
                lag_window,
                aggregation,
                ..
            } => Some((*lag_window, *aggregation)),
            _ => None,
        }
    }

    pub(crate) fn validate(&self) -> std::result::Result<(), InvalidTransformationDefinitionError> {
        if let Some((window, aggregation)) = self.timed_window() {
            timed::validate_window(window, aggregation)?;
        }
        if let Some((_, window, aggregation, _)) = self.statistic() {
            if window == 0 {
                return Err(InvalidTransformationDefinitionError::WindowZero);
            }
            statistics::validate_capacity(window, aggregation.map(|a| a.as_millis() as i64))?;
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
    let mut statistics =
        HashMap::<(usize, Option<i64>), Vec<(StatisticKind, u64, WarmupPolicy, usize)>>::new();
    let mut averages = HashMap::<(bool, usize, WarmupPolicy), (Vec<usize>, Vec<usize>)>::new();
    let mut timed_outputs = HashMap::<(usize, i64), Vec<(Option<ChangeKind>, i64, usize)>>::new();
    let mut changes = HashMap::<usize, Vec<(ChangeKind, usize, usize)>>::new();
    let mut lagged_outputs = HashMap::<usize, (Vec<usize>, Vec<usize>)>::new();
    for (output_index, definition) in definitions.iter().enumerate() {
        let input_index = inputs
            .iter()
            .position(|id| id == definition.input())
            .expect("pipeline construction validated every input ID");
        match definition {
            TransformerDefinition::RollingStd { .. }
            | TransformerDefinition::RollingStdTimed { .. }
            | TransformerDefinition::RollingMin { .. }
            | TransformerDefinition::RollingMinTimed { .. }
            | TransformerDefinition::RollingMax { .. }
            | TransformerDefinition::RollingMaxTimed { .. }
            | TransformerDefinition::RollingRange { .. }
            | TransformerDefinition::RollingRangeTimed { .. } => {
                let (kind, window, aggregation, warmup) = definition.statistic().unwrap();
                statistics
                    .entry((input_index, aggregation.map(|a| a.as_millis() as i64)))
                    .or_default()
                    .push((kind, window, warmup, output_index));
            }
            TransformerDefinition::LaggedTimed {
                lag_window,
                aggregation,
                ..
            }
            | TransformerDefinition::DeltaTimed {
                lag_window,
                aggregation,
                ..
            }
            | TransformerDefinition::SimpleReturnTimed {
                lag_window,
                aggregation,
                ..
            }
            | TransformerDefinition::LogReturnTimed {
                lag_window,
                aggregation,
                ..
            } => {
                let kind = match definition {
                    TransformerDefinition::DeltaTimed { .. } => Some(ChangeKind::Delta),
                    TransformerDefinition::SimpleReturnTimed { .. } => {
                        Some(ChangeKind::SimpleReturn)
                    }
                    TransformerDefinition::LogReturnTimed { .. } => Some(ChangeKind::LogReturn),
                    _ => None,
                };
                timed_outputs
                    .entry((input_index, aggregation.as_millis() as i64))
                    .or_default()
                    .push((kind, lag_window.as_millis() as i64, output_index));
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
    for ((input, aggregation), outputs) in timed_outputs {
        operations.push(Transformer::Timed(TimedTransformer::new(
            input,
            aggregation,
            outputs,
        )));
    }
    for ((input, aggregation), outputs) in statistics {
        operations.push(Transformer::Statistics(StatisticsTransformer::new(
            input,
            aggregation,
            outputs,
        )));
    }
    operations.into_boxed_slice()
}

/// Resolved scalar operation for allocation-free writes into model input.
pub(crate) enum Transformer {
    Statistics(StatisticsTransformer),
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
            Self::Statistics(transformer) => transformer.apply(
                raw_values,
                observed,
                model_vector,
                output_observed,
                evaluation,
            ),
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

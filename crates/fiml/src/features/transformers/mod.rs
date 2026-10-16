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
mod history;
mod identity;
mod lagged;
mod standard_scale;
mod state;
use state::{StateKind, StateTransformer};
mod trajectory;
use trajectory::{TrajectoryKind, TrajectoryTransformer};
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
/// Input and output IDs are stored once alongside the operation's parameters.
/// Convenience constructors and direct construction describe the same definition:
///
/// ```
/// use fiml::{FeatureId, TransformerDefinition, TransformerOperation};
///
/// let definition = TransformerDefinition {
///     input: FeatureId::new("price"),
///     output: FeatureId::new("previous_price"),
///     operation: TransformerOperation::Lagged { lag_window: 2 },
/// };
/// assert_eq!(definition, TransformerDefinition::lagged(
///     FeatureId::new("price"), FeatureId::new("previous_price"), 2,
/// ));
/// assert_eq!(definition.input.as_str(), "price");
/// assert!(matches!(definition.operation, TransformerOperation::Lagged { lag_window: 2 }));
/// ```
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
///
/// Trajectory transformers use observation-count windows and a separate shared tree
/// per input. Windows count finite observations, skipping nonfinite inputs while
/// emitting NaN until the next finite observation. Unobserved inputs retain outputs.
/// Full-window warmup requires `window` finite observations; first-value warmup
/// exposes partial windows. Slope requires two observations, even for `window = 1`.
/// Other trajectory measurements are zero for a singleton. Nonfinite arithmetic
/// results emit NaN and recover when the contributing observations leave the window.
/// Updates and distinct-window queries cost `O(log max_window)` without allocation.
/// Trajectory storage uses `2 * next_power_of_two(max_window)` summaries (96 bytes
/// each on 64-bit targets), shared across kinds, windows, and warmup policies.
///
/// State transformers count finite signed observations or milliseconds since a matching
/// signal. The inclusive interval `[-epsilon, epsilon]` is neutral and resets streaks.
/// Nonfinite observations preserve state and emit NaN until the next finite input;
/// unobserved inputs retain count outputs. Since-values are NaN until a match, then zero
/// on each match. Counts saturate at 2^53. Time ages use source clocks, refresh on clock
/// advances, ignore context updates, and use widened timestamp differences before f64
/// conversion. State updates use constant space and allocate nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerDefinition {
    /// ID to read from the preceding stage's layout.
    pub input: FeatureId,
    /// Unique, non-reserved ID assigned to this output.
    pub output: FeatureId,
    /// Calculation and parameters applied to the input.
    pub operation: TransformerOperation,
}

/// Scalar calculation parameters, independent of input and output IDs.
///
/// Used by [`TransformerDefinition`] to keep layout IDs in one place while
/// retaining operation-specific configuration. See that type for history and clock behavior.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerOperation {
    /// Consecutive observations greater than epsilon.
    PositiveStreak {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Consecutive observations less than negative epsilon.
    NegativeStreak {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Finite observations since the last value greater than epsilon.
    ObservationsSincePositive {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Finite observations since the last value less than negative epsilon.
    ObservationsSinceNegative {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Finite observations since the last value outside the neutral interval.
    ObservationsSinceNonzero {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Elapsed milliseconds since the last value greater than epsilon.
    TimeSincePositive {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Elapsed milliseconds since the last value less than negative epsilon.
    TimeSinceNegative {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Elapsed milliseconds since the last value outside the neutral interval.
    TimeSinceNonzero {
        /// Finite, nonnegative half-width of the inclusive neutral interval.
        epsilon: f64,
    },
    /// Least-squares slope against sample indices; NaN with fewer than two observations.
    RollingSlope {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Sum of absolute adjacent changes within the window.
    RollingPathLength {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Sum of positive adjacent changes within the window.
    RollingPositiveMovement {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Magnitude of negative adjacent changes within the window.
    RollingNegativeMovement {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Number of reversals between nonzero deltas; zero deltas are ignored.
    RollingDirectionChangeCount {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Largest earlier-peak to later-trough decrease, in absolute input units.
    RollingMaxDrawdown {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },

    /// Rolling population standard deviation over finite source observations.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingStd {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling population standard deviation over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingStdTimed {
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
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling minimum over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingMinTimed {
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
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling maximum over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingMaxTimed {
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
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// Rolling range (maximum minus minimum) over all finite observations in elapsed-time buckets.
    /// Missing observations emit NaN without entering history; compatible outputs share storage.
    RollingRangeTimed {
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
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },
    /// Emits current minus the as-of value at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    DeltaTimed {
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },
    /// Emits current / as-of value - 1 at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    SimpleReturnTimed {
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },
    /// Emits ln(current) - ln(as-of value) at a positive elapsed-time lag.
    /// Only finite observations enter shared history. Nonfinite observations emit NaN.
    /// Source-clock advances refresh the output and mark it observed for later stages.
    LogReturnTimed {
        /// Positive whole-millisecond duration, a multiple of `aggregation`.
        lag_window: Duration,
        /// Positive whole-millisecond bucket width. Each bucket retains its last finite observation.
        aggregation: Duration,
    },

    /// EMA over finite source observations; missing observations emit NaN without advancing history.
    Ema {
        /// Positive number of finite observations.
        window: usize,
        /// Controls availability while history fills.
        warmup_policy: WarmupPolicy,
    },
    /// SMA over finite source observations; missing observations emit NaN without advancing history.
    Sma {
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
        /// Number of finite observations to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Computes `current / previous - 1` over finite source observations.
    /// A zero denominator emits NaN.
    /// A lag of `k` requires `k + 1` observations; until then the output is NaN.
    /// Observed nonfinite values emit NaN without advancing history; unrelated events retain output.
    /// All change kinds and windows for the same input share history within a stage.
    SimpleReturn {
        /// Number of finite observations to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Computes `ln(current) - ln(previous)` over finite source observations.
    /// Nonpositive operands emit NaN.
    /// A lag of `k` requires `k + 1` observations; until then the output is NaN.
    /// Observed nonfinite values emit NaN without advancing history; unrelated events retain output.
    /// All change kinds and windows for the same input share history within a stage.
    LogReturn {
        /// Number of finite observations to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Copies one input scalar without changing its value.
    Identity,
    /// Emits the input scalar from `lag_window` accepted events earlier.
    /// The window must be in `1..=10_000`; output remains NaN until enough history exists.
    /// Definitions for the same input share one runtime history buffer.
    Lagged {
        /// Number of accepted events to look back, in `1..=10_000`.
        lag_window: usize,
    },
    /// Applies `(input - mean) / scale` to one input scalar.
    StandardScale {
        /// Finite centering value.
        mean: f64,
        /// Positive finite divisor whose reciprocal must also be finite.
        scale: f64,
    },
}

impl TransformerDefinition {
    /// Consecutive observations greater than epsilon.
    pub fn positive_streak(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::PositiveStreak { epsilon },
        }
    }

    /// Consecutive observations less than negative epsilon.
    pub fn negative_streak(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::NegativeStreak { epsilon },
        }
    }

    /// Finite observations since the last value greater than epsilon.
    pub fn observations_since_positive(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::ObservationsSincePositive { epsilon },
        }
    }

    /// Finite observations since the last value less than negative epsilon.
    pub fn observations_since_negative(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::ObservationsSinceNegative { epsilon },
        }
    }

    /// Finite observations since the last value outside the neutral interval.
    pub fn observations_since_nonzero(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::ObservationsSinceNonzero { epsilon },
        }
    }

    /// Elapsed milliseconds since the last value greater than epsilon.
    pub fn time_since_positive(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::TimeSincePositive { epsilon },
        }
    }

    /// Elapsed milliseconds since the last value less than negative epsilon.
    pub fn time_since_negative(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::TimeSinceNegative { epsilon },
        }
    }

    /// Elapsed milliseconds since the last value outside the neutral interval.
    pub fn time_since_nonzero(input: FeatureId, output: FeatureId, epsilon: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::TimeSinceNonzero { epsilon },
        }
    }

    fn state(&self) -> Option<(StateKind, f64)> {
        match &self.operation {
            TransformerOperation::PositiveStreak { epsilon } => {
                Some((StateKind::PositiveStreak, *epsilon))
            }
            TransformerOperation::NegativeStreak { epsilon } => {
                Some((StateKind::NegativeStreak, *epsilon))
            }
            TransformerOperation::ObservationsSincePositive { epsilon } => {
                Some((StateKind::ObservationsSincePositive, *epsilon))
            }
            TransformerOperation::ObservationsSinceNegative { epsilon } => {
                Some((StateKind::ObservationsSinceNegative, *epsilon))
            }
            TransformerOperation::ObservationsSinceNonzero { epsilon } => {
                Some((StateKind::ObservationsSinceNonzero, *epsilon))
            }
            TransformerOperation::TimeSincePositive { epsilon } => {
                Some((StateKind::TimeSincePositive, *epsilon))
            }
            TransformerOperation::TimeSinceNegative { epsilon } => {
                Some((StateKind::TimeSinceNegative, *epsilon))
            }
            TransformerOperation::TimeSinceNonzero { epsilon } => {
                Some((StateKind::TimeSinceNonzero, *epsilon))
            }
            _ => None,
        }
    }

    pub(crate) fn needs_clock(&self) -> bool {
        self.timed_window().is_some() || self.state().is_some_and(|(kind, _)| kind.is_timed())
    }

    /// Least-squares slope against sample indices; NaN with fewer than two observations.
    pub fn rolling_slope(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingSlope {
                window,
                warmup_policy,
            },
        }
    }
    /// Sum of absolute adjacent changes within the window.
    pub fn rolling_path_length(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingPathLength {
                window,
                warmup_policy,
            },
        }
    }
    /// Sum of positive adjacent changes within the window.
    pub fn rolling_positive_movement(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingPositiveMovement {
                window,
                warmup_policy,
            },
        }
    }
    /// Magnitude of negative adjacent changes within the window.
    pub fn rolling_negative_movement(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingNegativeMovement {
                window,
                warmup_policy,
            },
        }
    }
    /// Number of reversals between nonzero deltas; zero deltas are ignored.
    pub fn rolling_direction_change_count(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingDirectionChangeCount {
                window,
                warmup_policy,
            },
        }
    }
    /// Largest earlier-peak to later-trough decrease, in absolute input units.
    pub fn rolling_max_drawdown(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingMaxDrawdown {
                window,
                warmup_policy,
            },
        }
    }
    fn trajectory(&self) -> Option<(TrajectoryKind, usize, WarmupPolicy)> {
        match &self.operation {
            TransformerOperation::RollingSlope {
                window,
                warmup_policy,
            } => Some((TrajectoryKind::Slope, *window, *warmup_policy)),
            TransformerOperation::RollingPathLength {
                window,
                warmup_policy,
            } => Some((TrajectoryKind::PathLength, *window, *warmup_policy)),
            TransformerOperation::RollingPositiveMovement {
                window,
                warmup_policy,
            } => Some((TrajectoryKind::PositiveMovement, *window, *warmup_policy)),
            TransformerOperation::RollingNegativeMovement {
                window,
                warmup_policy,
            } => Some((TrajectoryKind::NegativeMovement, *window, *warmup_policy)),
            TransformerOperation::RollingDirectionChangeCount {
                window,
                warmup_policy,
            } => Some((
                TrajectoryKind::DirectionChangeCount,
                *window,
                *warmup_policy,
            )),
            TransformerOperation::RollingMaxDrawdown {
                window,
                warmup_policy,
            } => Some((TrajectoryKind::MaxDrawdown, *window, *warmup_policy)),
            _ => None,
        }
    }

    /// Creates an observation-based rolling population standard deviation.
    pub fn rolling_std(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingStd {
                window,
                warmup_policy,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::RollingStdTimed {
                window,
                aggregation,
                warmup_policy,
            },
        }
    }
    /// Creates an observation-based rolling minimum.
    pub fn rolling_min(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingMin {
                window,
                warmup_policy,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::RollingMinTimed {
                window,
                aggregation,
                warmup_policy,
            },
        }
    }
    /// Creates an observation-based rolling maximum.
    pub fn rolling_max(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingMax {
                window,
                warmup_policy,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::RollingMaxTimed {
                window,
                aggregation,
                warmup_policy,
            },
        }
    }
    /// Creates an observation-based rolling range (maximum minus minimum).
    pub fn rolling_range(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::RollingRange {
                window,
                warmup_policy,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::RollingRangeTimed {
                window,
                aggregation,
                warmup_policy,
            },
        }
    }
    fn statistic(&self) -> Option<(StatisticKind, u64, Option<Duration>, WarmupPolicy)> {
        match &self.operation {
            TransformerOperation::RollingStd {
                window,
                warmup_policy,
            } => Some((StatisticKind::Std, *window as u64, None, *warmup_policy)),
            TransformerOperation::RollingStdTimed {
                window,
                warmup_policy,
                aggregation,
            } => Some((
                StatisticKind::Std,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            TransformerOperation::RollingMin {
                window,
                warmup_policy,
            } => Some((StatisticKind::Min, *window as u64, None, *warmup_policy)),
            TransformerOperation::RollingMinTimed {
                window,
                warmup_policy,
                aggregation,
            } => Some((
                StatisticKind::Min,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            TransformerOperation::RollingMax {
                window,
                warmup_policy,
            } => Some((StatisticKind::Max, *window as u64, None, *warmup_policy)),
            TransformerOperation::RollingMaxTimed {
                window,
                warmup_policy,
                aggregation,
            } => Some((
                StatisticKind::Max,
                window.as_millis() as u64,
                Some(*aggregation),
                *warmup_policy,
            )),
            TransformerOperation::RollingRange {
                window,
                warmup_policy,
            } => Some((StatisticKind::Range, *window as u64, None, *warmup_policy)),
            TransformerOperation::RollingRangeTimed {
                window,
                warmup_policy,
                aggregation,
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
        Self {
            input,
            output,
            operation: TransformerOperation::LaggedTimed {
                lag_window,
                aggregation,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::DeltaTimed {
                lag_window,
                aggregation,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::SimpleReturnTimed {
                lag_window,
                aggregation,
            },
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
        Self {
            input,
            output,
            operation: TransformerOperation::LogReturnTimed {
                lag_window,
                aggregation,
            },
        }
    }

    /// Creates a sample-based EMA transformation.
    pub fn ema(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::Ema {
                window,
                warmup_policy,
            },
        }
    }

    /// Creates a sample-based SMA transformation.
    pub fn sma(
        input: FeatureId,
        output: FeatureId,
        window: usize,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::Sma {
                window,
                warmup_policy,
            },
        }
    }

    /// Creates an exact observation-based delta transformation.
    pub fn delta(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::Delta { lag_window },
        }
    }

    /// Creates an exact observation-based simple return transformation.
    pub fn simple_return(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::SimpleReturn { lag_window },
        }
    }

    /// Creates an exact observation-based log return transformation.
    pub fn log_return(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::LogReturn { lag_window },
        }
    }

    /// Creates a scalar identity transformation.
    pub fn identity(input: FeatureId, output: FeatureId) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::Identity,
        }
    }

    /// Creates an event-lagged transformation with a positive history window.
    pub fn lagged(input: FeatureId, output: FeatureId, lag_window: usize) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::Lagged { lag_window },
        }
    }

    /// Creates a scalar standard-scaling transformation.
    pub fn standard_scale(input: FeatureId, output: FeatureId, mean: f64, scale: f64) -> Self {
        Self {
            input,
            output,
            operation: TransformerOperation::StandardScale { mean, scale },
        }
    }

    pub(crate) fn timed_window(&self) -> Option<(Duration, Duration)> {
        match &self.operation {
            TransformerOperation::RollingStdTimed {
                window,
                aggregation,
                ..
            }
            | TransformerOperation::RollingMinTimed {
                window,
                aggregation,
                ..
            }
            | TransformerOperation::RollingMaxTimed {
                window,
                aggregation,
                ..
            }
            | TransformerOperation::RollingRangeTimed {
                window,
                aggregation,
                ..
            } => Some((*window, *aggregation)),
            TransformerOperation::LaggedTimed {
                lag_window,
                aggregation,
            }
            | TransformerOperation::DeltaTimed {
                lag_window,
                aggregation,
            }
            | TransformerOperation::SimpleReturnTimed {
                lag_window,
                aggregation,
            }
            | TransformerOperation::LogReturnTimed {
                lag_window,
                aggregation,
            } => Some((*lag_window, *aggregation)),
            _ => None,
        }
    }

    pub(crate) fn validate(&self) -> std::result::Result<(), InvalidTransformationDefinitionError> {
        if let Some((_, epsilon)) = self.state()
            && (!epsilon.is_finite() || epsilon < 0.)
        {
            return Err(InvalidTransformationDefinitionError::InvalidEpsilon);
        }
        if let Some((window, aggregation)) = self.timed_window() {
            timed::validate_window(window, aggregation)?;
        }
        if let Some((_, window, _)) = self.trajectory() {
            trajectory::validate_capacity(window)?;
        }
        if let Some((_, window, aggregation, _)) = self.statistic() {
            if window == 0 {
                return Err(InvalidTransformationDefinitionError::WindowZero);
            }
            statistics::validate_capacity(window, aggregation.map(|a| a.as_millis() as i64))?;
        }
        if matches!(
            &self.operation,
            TransformerOperation::Sma { window: 0, .. }
                | TransformerOperation::Ema { window: 0, .. }
        ) {
            return Err(InvalidTransformationDefinitionError::WindowZero);
        }
        if let TransformerOperation::Lagged { lag_window }
        | TransformerOperation::Delta { lag_window }
        | TransformerOperation::SimpleReturn { lag_window }
        | TransformerOperation::LogReturn { lag_window } = &self.operation
        {
            if *lag_window == 0 {
                return Err(InvalidTransformationDefinitionError::LagWindowZero);
            }
            if *lag_window > WINDOW_MAX_SIZE {
                return Err(InvalidTransformationDefinitionError::LagWindowTooLarge);
            }
        }
        if let TransformerOperation::StandardScale { mean, scale } = &self.operation {
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
        let reason = if !inputs.contains(&definition.input) {
            Some(InvalidTransformationDefinitionError::InputFeatureNotFound)
        } else if crate::features::is_reserved_feature_id(&definition.output) {
            Some(InvalidTransformationDefinitionError::ReservedOutputFeature)
        } else if definitions[..index]
            .iter()
            .any(|previous| previous.output == definition.output)
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
    match &definition.operation {
        TransformerOperation::Sma { warmup_policy, .. } => {
            Some((false, &definition.input, *warmup_policy))
        }
        TransformerOperation::Ema { warmup_policy, .. } => {
            Some((true, &definition.input, *warmup_policy))
        }
        _ => None,
    }
}

/// Compiles validated scalar definitions, sharing history between lags of one input.
pub(crate) fn compile(
    definitions: &[TransformerDefinition],
    inputs: &[FeatureId],
) -> Box<[Transformer]> {
    let mut operations = Vec::with_capacity(definitions.len());
    let mut trajectories =
        HashMap::<usize, Vec<(TrajectoryKind, usize, WarmupPolicy, usize)>>::new();
    let mut statistics =
        HashMap::<(usize, Option<i64>), Vec<(StatisticKind, u64, WarmupPolicy, usize)>>::new();
    let mut averages = HashMap::<(bool, usize, WarmupPolicy), (Vec<usize>, Vec<usize>)>::new();
    let mut timed_outputs = HashMap::<(usize, i64), Vec<(Option<ChangeKind>, i64, usize)>>::new();
    let mut changes = HashMap::<usize, Vec<(ChangeKind, usize, usize)>>::new();
    let mut lagged_outputs = HashMap::<usize, (Vec<usize>, Vec<usize>)>::new();
    for (output_index, definition) in definitions.iter().enumerate() {
        let input_index = inputs
            .iter()
            .position(|id| id == &definition.input)
            .expect("pipeline construction validated every input ID");
        match &definition.operation {
            TransformerOperation::PositiveStreak { .. }
            | TransformerOperation::NegativeStreak { .. }
            | TransformerOperation::ObservationsSincePositive { .. }
            | TransformerOperation::ObservationsSinceNegative { .. }
            | TransformerOperation::ObservationsSinceNonzero { .. }
            | TransformerOperation::TimeSincePositive { .. }
            | TransformerOperation::TimeSinceNegative { .. }
            | TransformerOperation::TimeSinceNonzero { .. } => {
                let (kind, epsilon) = definition.state().unwrap();
                operations.push(Transformer::State(StateTransformer::new(
                    input_index,
                    output_index,
                    kind,
                    epsilon,
                )));
            }
            TransformerOperation::RollingSlope { .. }
            | TransformerOperation::RollingPathLength { .. }
            | TransformerOperation::RollingPositiveMovement { .. }
            | TransformerOperation::RollingNegativeMovement { .. }
            | TransformerOperation::RollingDirectionChangeCount { .. }
            | TransformerOperation::RollingMaxDrawdown { .. } => {
                let (kind, window, warmup) = definition.trajectory().unwrap();
                trajectories.entry(input_index).or_default().push((
                    kind,
                    window,
                    warmup,
                    output_index,
                ));
            }
            TransformerOperation::RollingStd { .. }
            | TransformerOperation::RollingStdTimed { .. }
            | TransformerOperation::RollingMin { .. }
            | TransformerOperation::RollingMinTimed { .. }
            | TransformerOperation::RollingMax { .. }
            | TransformerOperation::RollingMaxTimed { .. }
            | TransformerOperation::RollingRange { .. }
            | TransformerOperation::RollingRangeTimed { .. } => {
                let (kind, window, aggregation, warmup) = definition.statistic().unwrap();
                statistics
                    .entry((input_index, aggregation.map(|a| a.as_millis() as i64)))
                    .or_default()
                    .push((kind, window, warmup, output_index));
            }
            TransformerOperation::LaggedTimed {
                lag_window,
                aggregation,
            }
            | TransformerOperation::DeltaTimed {
                lag_window,
                aggregation,
            }
            | TransformerOperation::SimpleReturnTimed {
                lag_window,
                aggregation,
            }
            | TransformerOperation::LogReturnTimed {
                lag_window,
                aggregation,
            } => {
                let kind = match &definition.operation {
                    TransformerOperation::DeltaTimed { .. } => Some(ChangeKind::Delta),
                    TransformerOperation::SimpleReturnTimed { .. } => {
                        Some(ChangeKind::SimpleReturn)
                    }
                    TransformerOperation::LogReturnTimed { .. } => Some(ChangeKind::LogReturn),
                    _ => None,
                };
                timed_outputs
                    .entry((input_index, aggregation.as_millis() as i64))
                    .or_default()
                    .push((kind, lag_window.as_millis() as i64, output_index));
            }
            TransformerOperation::Sma { window, .. } | TransformerOperation::Ema { window, .. } => {
                let (ema, _, warmup) = average_key(definition).unwrap();
                let (windows, outputs) = averages.entry((ema, input_index, warmup)).or_default();
                windows.push(*window);
                outputs.push(output_index);
            }
            TransformerOperation::Delta { lag_window }
            | TransformerOperation::SimpleReturn { lag_window }
            | TransformerOperation::LogReturn { lag_window } => {
                let kind = match &definition.operation {
                    TransformerOperation::Delta { .. } => ChangeKind::Delta,
                    TransformerOperation::SimpleReturn { .. } => ChangeKind::SimpleReturn,
                    _ => ChangeKind::LogReturn,
                };
                changes
                    .entry(input_index)
                    .or_default()
                    .push((kind, *lag_window, output_index));
            }
            TransformerOperation::Identity => operations.push(Transformer::Identity(
                IdentityTransformer::new(input_index, output_index),
            )),
            TransformerOperation::Lagged { lag_window } => {
                let (windows, output_indices) = lagged_outputs.entry(input_index).or_default();
                windows.push(*lag_window);
                output_indices.push(output_index);
            }
            TransformerOperation::StandardScale { mean, scale } => {
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
    for (input, outputs) in trajectories {
        operations.push(Transformer::Trajectory(TrajectoryTransformer::new(
            input, outputs,
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
    State(StateTransformer),
    Trajectory(TrajectoryTransformer),
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
            Self::State(transformer) => transformer.apply(
                raw_values,
                observed,
                model_vector,
                output_observed,
                evaluation,
            ),
            Self::Trajectory(transformer) => {
                transformer.apply(raw_values, observed, model_vector, output_observed)
            }
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
            if matches!(&definition.operation, TransformerOperation::Lagged { .. }) {
                None
            } else {
                Some(
                    inputs
                        .iter()
                        .position(|id| id == &definition.input)
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

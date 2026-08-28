pub mod event;
pub mod features;
pub mod indicators;
pub mod order_book;
mod ring_buffer;
pub mod symbols;
mod types;
mod vectors;

use std::{error::Error, fmt::Display, time::Duration};

use rust_decimal::Decimal;

pub use event::{
    EVENT_KIND_COUNT, Event, EventKind, OrderBookDeltaEvent, OrderBookSnapshotEvent, PriceUpdate,
    TimeUpdate, TradeSide, TradeUpdate, VolumeUpdate,
};
pub use features::{
    EventField, FeatureDefinition, FeatureExtractor, FeatureExtractorBuilder, FeatureId,
    FeatureKey, FeatureSource, FeatureVectorSpec, MAX_OUTPUTS_PER_INDICATOR, UpdateResult,
};
pub use indicators::{CumulativeVolumeDelta, ObvBucket, OnBalanceVolumeTimed};
pub use ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
pub use symbols::Symbol;
pub use types::{Float, WarmupPolicy};
pub use vectors::{ArrayFeatureVector, FeatureVector, VecFeatureVector};

use crate::order_book::OrderBookUpdateError;

pub type Result<T> = std::result::Result<T, FimlError>;

#[derive(Debug)]
#[non_exhaustive]
pub enum FimlError {
    InvalidArgument(InvalidArgumentError),
    InvalidPriceRange {
        from_price: Decimal,
        to_price: Decimal,
    },
    InvalidIndicatorDefinition {
        index: usize,
        indicator: IndicatorKind,
        reason: InvalidIndicatorDefinitionError,
    },
    OutputCountMismatch {
        expected: usize,
        actual: usize,
    },
    FeatureVectorCapacityMismatch {
        expected: usize,
        actual: usize,
    },
    TimestampOutOfOrder {
        symbol: Symbol,
        event_kind: EventKind,
        timestamp: i64,
        previous_timestamp: i64,
    },
    OrderBookUpdateError {
        reason: OrderBookUpdateError,
    },
    OrderBookNotConfigured {
        symbol: Symbol,
    },
    DuplicateOrderBook {
        symbol: Symbol,
    },
}

/// Allocation-free details for invalid public API arguments.
///
/// The variants retain the values needed to diagnose a failure without building an owned error
/// message when the error is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidArgumentError {
    FeatureVectorIndexOutOfBounds {
        index: usize,
        length: usize,
    },
    SourceValuesTooShort {
        requested: usize,
        available: usize,
    },
    FeatureVectorRangeOverflow {
        start: usize,
        size: usize,
    },
    FeatureVectorRangeOutOfBounds {
        start: usize,
        end: usize,
        capacity: usize,
    },
    LimitExceeded {
        target: LimitTarget,
        count: usize,
        limit: usize,
    },
    FeatureVectorCapacityTooSmall {
        capacity: usize,
        active_length: usize,
    },
    ReservedFeatureId {
        definition_index: usize,
    },
    RingBufferCapacityZero,
    WindowLimitReached {
        limit: usize,
    },
    WindowAddedAfterData,
    WindowPeriodZero,
    WindowPeriodExceedsCapacity {
        period: usize,
        capacity: usize,
    },
    WindowPeriodMustBeLessThanCapacity {
        period: usize,
        capacity: usize,
    },
    AggregationTooShort,
    DurationPrecision {
        field: DurationField,
    },
    DurationOutOfRange {
        field: DurationField,
    },
    WindowShorterThanAggregation,
    WindowNotMultipleOfAggregation,
    WindowPeriodOutOfRange {
        target: IntegerTarget,
    },
    WindowDurationOutOfRange,
    TimedPeriodTooLarge {
        indicator: IndicatorKind,
    },
}

/// Collection whose fixed-size representation was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LimitTarget {
    RuntimeFeatures,
    SymbolRouters,
    Subscribers,
    SubscriberGroup,
    OrderBooks,
    Transformers,
}

/// Duration argument involved in validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DurationField {
    Aggregation,
    Window,
    TimedWindow,
}

/// Integer representation required by an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegerTarget {
    Signed64,
    Usize,
}

/// Indicator family associated with a compiled definition or construction error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndicatorKind {
    Sma,
    Ema,
    Cvd,
    SmaTimed,
    ObvTimed,
    TradeCountTimed,
    DayOfWeek,
    TimeSinceFirstEventOfDay,
}

/// Allocation-free reason why a feature definition could not be compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidIndicatorDefinitionError {
    CompatibleGroupOutputLimitExceeded {
        limit: usize,
    },
    DuplicateScalarDerivation,
    DuplicateFeatureKey,
    DuplicateFeatureId,
    ScalarEventFieldSourceRequired,
    TradeEventSourceRequired,
    WindowTooShort,
    WindowTooLarge,
    AggregationTooShort,
    WindowShorterThanAggregation {
        aggregation_millis: i64,
        window_millis: i64,
    },
    WindowNotMultipleOfAggregation {
        aggregation_millis: i64,
        window_millis: i64,
    },
    BucketPeriodOutOfRange,
    DurationPrecision {
        field: DefinitionDurationField,
        duration: Duration,
    },
    DurationOutOfRange {
        field: DefinitionDurationField,
        duration: Duration,
    },
    UtcOffsetOutOfRange {
        offset_millis: i64,
    },
    UtcOffsetPrecision {
        offset_millis: i64,
    },
    InvalidArgument(InvalidArgumentError),
    ConstructionFailed,
}

/// Duration field in an invalid compiled feature definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DefinitionDurationField {
    Aggregation,
    Window,
}

impl Display for FimlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FimlError::InvalidArgument(reason) => write!(f, "invalid argument: {reason}"),
            FimlError::InvalidPriceRange {
                from_price,
                to_price,
            } => write!(
                f,
                "invalid price range: from price {from_price} must be less than to price {to_price}"
            ),
            FimlError::InvalidIndicatorDefinition {
                index,
                indicator,
                reason,
            } => {
                write!(
                    f,
                    "invalid indicator definition at index {index}: {indicator}: {reason}"
                )
            }
            FimlError::OutputCountMismatch { expected, actual } => {
                write!(
                    f,
                    "output storage has {actual} cells, but compilation requires exactly {expected}"
                )
            }
            FimlError::FeatureVectorCapacityMismatch { expected, actual } => write!(
                f,
                "output storage has capacity {actual}, but the feature-vector spec requires capacity {expected}"
            ),
            FimlError::TimestampOutOfOrder {
                symbol,
                event_kind,
                timestamp,
                previous_timestamp,
            } => {
                write!(f, "timestamp {timestamp} for {event_kind}")?;
                write!(f, " event for symbol {}", symbol)?;
                write!(
                    f,
                    " is earlier than previous timestamp {previous_timestamp}"
                )
            }
            FimlError::OrderBookUpdateError { reason } => {
                write!(f, "order-book update failed: {reason}")
            }
            FimlError::OrderBookNotConfigured { symbol } => {
                write!(f, "no order book is configured for symbol {symbol}")
            }
            FimlError::DuplicateOrderBook { symbol } => {
                write!(
                    f,
                    "more than one order book is configured for symbol {symbol}"
                )
            }
        }
    }
}

impl Error for FimlError {}

impl Display for InvalidArgumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FeatureVectorIndexOutOfBounds { index, length } => write!(
                f,
                "index {index} is out of bounds for feature vector of length {length}"
            ),
            Self::SourceValuesTooShort {
                requested,
                available,
            } => write!(
                f,
                "requested size {requested} exceeds the {available} provided values"
            ),
            Self::FeatureVectorRangeOverflow { start, size } => {
                write!(
                    f,
                    "range starting at {start} with size {size} overflows usize"
                )
            }
            Self::FeatureVectorRangeOutOfBounds {
                start,
                end,
                capacity,
            } => write!(
                f,
                "range {start}..{end} is out of bounds for feature vector capacity {capacity}"
            ),
            Self::LimitExceeded {
                target,
                count,
                limit,
            } => write!(f, "{target} count {count} exceeds limit {limit}"),
            Self::FeatureVectorCapacityTooSmall {
                capacity,
                active_length,
            } => write!(
                f,
                "feature vector capacity {capacity} is smaller than active length {active_length}"
            ),
            Self::ReservedFeatureId { definition_index } => write!(
                f,
                "feature definition at index {definition_index} uses the reserved namespace for feature IDs"
            ),
            Self::RingBufferCapacityZero => {
                f.write_str("ring buffer capacity must be greater than 0")
            }
            Self::WindowLimitReached { limit } => {
                write!(f, "maximum number of windows ({limit}) reached")
            }
            Self::WindowAddedAfterData => {
                f.write_str("cannot add a window after data has been added")
            }
            Self::WindowPeriodZero => f.write_str("window period must be greater than 0"),
            Self::WindowPeriodExceedsCapacity { period, capacity } => write!(
                f,
                "window period {period} cannot exceed ring buffer capacity {capacity}"
            ),
            Self::WindowPeriodMustBeLessThanCapacity { period, capacity } => write!(
                f,
                "window period {period} must be less than ring buffer capacity {capacity}"
            ),
            Self::AggregationTooShort => {
                f.write_str("aggregation duration must be at least 1 millisecond")
            }
            Self::DurationPrecision { field } => {
                write!(f, "{field} must use whole-millisecond precision")
            }
            Self::DurationOutOfRange { field } => {
                write!(f, "{field} must fit signed 64-bit milliseconds")
            }
            Self::WindowShorterThanAggregation => {
                f.write_str("window cannot be shorter than aggregation")
            }
            Self::WindowNotMultipleOfAggregation => {
                f.write_str("window must be a multiple of aggregation")
            }
            Self::WindowPeriodOutOfRange { target } => {
                write!(f, "window period must fit {target}")
            }
            Self::WindowDurationOutOfRange => {
                f.write_str("window duration must fit signed 64-bit milliseconds")
            }
            Self::TimedPeriodTooLarge { indicator } => {
                write!(f, "{indicator} timed period is too large")
            }
        }
    }
}

impl Display for LimitTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RuntimeFeatures => "runtime feature",
            Self::SymbolRouters => "symbol router",
            Self::Subscribers => "subscriber",
            Self::SubscriberGroup => "subscriber group",
            Self::OrderBooks => "order-book",
            Self::Transformers => "transformer",
        })
    }
}

impl Display for DurationField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Aggregation => "aggregation duration",
            Self::Window => "window duration",
            Self::TimedWindow => "timed window durations",
        })
    }
}

impl Display for IntegerTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Signed64 => "signed 64-bit",
            Self::Usize => "usize",
        })
    }
}

impl Display for IndicatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Sma => "SMA",
            Self::Ema => "EMA",
            Self::Cvd => "CVD",
            Self::SmaTimed => "timed SMA",
            Self::ObvTimed => "timed OBV",
            Self::TradeCountTimed => "timed trade count",
            Self::DayOfWeek => "day of week",
            Self::TimeSinceFirstEventOfDay => "time since first event of day",
        })
    }
}

impl Display for InvalidIndicatorDefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompatibleGroupOutputLimitExceeded { limit } => {
                write!(f, "compatible feature group exceeds {limit} outputs")
            }
            Self::DuplicateScalarDerivation => {
                f.write_str("duplicates a scalar runtime derivation")
            }
            Self::DuplicateFeatureKey => f.write_str("duplicates an earlier feature key"),
            Self::DuplicateFeatureId => f.write_str("duplicates an earlier feature ID"),
            Self::ScalarEventFieldSourceRequired => {
                f.write_str("requires a scalar event-field source")
            }
            Self::TradeEventSourceRequired => f.write_str("requires Event(Trade) as its source"),
            Self::WindowTooShort => f.write_str("window must be at least 1"),
            Self::WindowTooLarge => f.write_str("window is too large"),
            Self::AggregationTooShort => f.write_str("aggregation must be at least 1 millisecond"),
            Self::WindowShorterThanAggregation {
                aggregation_millis,
                window_millis,
            } => write!(
                f,
                "window must be at least aggregation {aggregation_millis}ms, got {window_millis}ms"
            ),
            Self::WindowNotMultipleOfAggregation {
                aggregation_millis,
                window_millis,
            } => write!(
                f,
                "window must be an exact multiple of aggregation {aggregation_millis}ms, got {window_millis}ms"
            ),
            Self::BucketPeriodOutOfRange => f.write_str("derived bucket period does not fit usize"),
            Self::DurationPrecision { field, duration } => write!(
                f,
                "{field} must use whole-millisecond precision, got {duration:?}"
            ),
            Self::DurationOutOfRange { field, duration } => write!(
                f,
                "{field} must fit signed 64-bit milliseconds, got {duration:?}"
            ),
            Self::UtcOffsetOutOfRange { offset_millis } => write!(
                f,
                "UTC offset must be within -14h..=+14h, got {offset_millis}ms"
            ),
            Self::UtcOffsetPrecision { offset_millis } => write!(
                f,
                "UTC offset must use whole-minute precision, got {offset_millis}ms"
            ),
            Self::InvalidArgument(reason) => write!(f, "invalid argument: {reason}"),
            Self::ConstructionFailed => f.write_str("indicator construction failed"),
        }
    }
}

impl Display for DefinitionDurationField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Aggregation => "aggregation",
            Self::Window => "window",
        })
    }
}

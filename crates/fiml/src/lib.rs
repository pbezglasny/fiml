//! Allocation-conscious streaming financial indicators and feature-vector pipelines for ML inference.
//!
//! Define features with [`FeatureKey`], build a [`FeatureExtractor`] with caller-provided
//! storage, and feed it market [`Event`]s. The extractor updates the same output storage
//! as events arrive; [`FeatureVector::values`] borrows its values as an `&[f64]` for
//! use by your model integration.
//!
//! # Example
//!
//! Extract price and compute a three-observation moving average. Construction allocates
//! history once; event processing writes into caller-owned storage.
//!
//! ```rust
//! use fiml::{ArrayFeatureVector, Event, EventField, FeatureDefinition,
//!     FeatureExtractorSpec, FeatureId, FeatureKey, PipelineSpec, Symbol,
//!     TransformerDefinition, WarmupPolicy};
//! # fn main() -> fiml::Result<()> {
//! let btc = Symbol::new("BTCUSDT")?;
//! let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
//!     FeatureKey::Field { symbol: btc, field: EventField::Price }, FeatureId::new("price"),
//! )])?;
//! let mut pipeline = PipelineSpec::new(raw, [TransformerDefinition::sma(
//!     FeatureId::new("price"), FeatureId::new("sma3"), 3, WarmupPolicy::FullWindow,
//! )])?.build(ArrayFeatureVector::<1>::new(), ArrayFeatureVector::<1>::new())?;
//! pipeline.handle_event(Event::price(btc, 10.0, 0))?;
//! assert!(pipeline.values()[0].is_nan());
//! pipeline.handle_event(Event::price(btc, 12.0, 1))?;
//! pipeline.handle_event(Event::price(btc, 14.0, 2))?;
//! assert_eq!(pipeline.values(), &[12.0]);
//! # Ok(())
//! # }
//! ```
//!
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
    Event, EventKind, OrderBookDeltaEvent, OrderBookSnapshotEvent, PriceUpdate, TimeUpdate,
    TradeSide, TradeUpdate, VolumeUpdate,
};
pub use features::{
    EventField, FeatureDefinition, FeatureExtractor, FeatureExtractorBuilder, FeatureExtractorSpec,
    FeatureId, FeatureKey, FeatureSource, FittedStage, MAX_OUTPUTS_PER_INDICATOR, Pipeline,
    PipelineSpec, TransformerDefinition, UpdateResult,
};
pub use indicators::{
    CumulativeVolumeDelta, ObvBucket, OnBalanceVolumeTimed, ReturnKind, VolumePriceTrend,
};
pub use ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
pub use symbols::Symbol;
pub use types::WarmupPolicy;
pub use vectors::{ArrayFeatureVector, FeatureVector, VecFeatureVector};

use crate::order_book::OrderBookUpdateError;

/// Result returned by fallible construction and event processing.
pub type Result<T> = std::result::Result<T, FimlError>;

/// Validation, construction, and event-processing failures.
#[derive(Debug)]
#[non_exhaustive]
pub enum FimlError {
    /// An atomic context update failed before any values were written.
    InvalidContextUpdate {
        /// Zero-based index in the supplied update slice.
        index: usize,
        /// Allocation-free diagnostic describing the invalid target or value.
        reason: &'static str,
    },
    /// An argument failed validation.
    InvalidArgument(InvalidArgumentError),
    /// The lower price bound is not below the upper bound.
    InvalidPriceRange {
        /// Requested lower price bound.
        from_price: Decimal,
        /// Requested upper price bound.
        to_price: Decimal,
    },
    /// A feature definition failed compilation or construction.
    InvalidIndicatorDefinition {
        /// Zero-based index of the invalid definition or stage.
        index: usize,
        /// Indicator family being configured.
        indicator: IndicatorKind,
        /// Underlying validation failure.
        reason: InvalidIndicatorDefinitionError,
    },
    /// A scalar transformation has invalid inputs or parameters.
    InvalidTransformationDefinition {
        /// Zero-based index of the invalid definition or stage.
        index: usize,
        /// Underlying validation failure.
        reason: InvalidTransformationDefinitionError,
    },
    /// Cold-path stage validation includes layout and numeric diagnostics.
    InvalidPipelineStage {
        /// Zero-based index of the invalid definition or stage.
        index: usize,
        /// Underlying validation failure.
        reason: String,
    },
    /// The output storage length differs from the compiled feature count.
    OutputCountMismatch {
        /// Required count or capacity.
        expected: usize,
        /// Supplied count or capacity.
        actual: usize,
    },
    /// The output capacity differs from the specification.
    FeatureVectorCapacityMismatch {
        /// Required count or capacity.
        expected: usize,
        /// Supplied count or capacity.
        actual: usize,
    },
    /// The model-input length differs from the pipeline specification.
    ModelVectorLengthMismatch {
        /// Required count or capacity.
        expected: usize,
        /// Supplied count or capacity.
        actual: usize,
    },
    /// The model-input capacity differs from the pipeline specification.
    ModelVectorCapacityMismatch {
        /// Required count or capacity.
        expected: usize,
        /// Supplied count or capacity.
        actual: usize,
    },
    /// An event predates the last accepted event for its symbol.
    TimestampOutOfOrder {
        /// Market symbol associated with the failure.
        symbol: Symbol,
        /// Kind of the rejected event.
        event_kind: EventKind,
        /// Rejected timestamp in epoch milliseconds.
        timestamp: i64,
        /// Last accepted timestamp for this symbol in epoch milliseconds.
        previous_timestamp: i64,
    },
    /// An order-book update failed validation or synchronization.
    OrderBookUpdateError {
        /// Underlying validation failure.
        reason: OrderBookUpdateError,
    },
    /// A book-dependent feature has no configured order book.
    OrderBookNotConfigured {
        /// Market symbol associated with the failure.
        symbol: Symbol,
    },
    /// Multiple order books were supplied for one symbol.
    DuplicateOrderBook {
        /// Market symbol associated with the failure.
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
    /// An order book was assigned the reserved global symbol.
    GlobalOrderBookSymbol,
    /// An event field contains NaN or infinity.
    NonFiniteEventValue {
        /// Argument or event field that failed validation.
        field: EventField,
    },
    /// A scalar write targets an index outside the active length.
    FeatureVectorIndexOutOfBounds {
        /// Zero-based destination index.
        index: usize,
        /// Active destination length.
        length: usize,
    },
    /// A range write requests more values than the source contains.
    SourceValuesTooShort {
        /// Number of source values requested.
        requested: usize,
        /// Number of values in the source slice.
        available: usize,
    },
    /// The range start plus its size overflows `usize`.
    FeatureVectorRangeOverflow {
        /// Inclusive destination start index.
        start: usize,
        /// Number of cells to write.
        size: usize,
    },
    /// A range write exceeds the destination capacity.
    FeatureVectorRangeOutOfBounds {
        /// Inclusive destination start index.
        start: usize,
        /// Exclusive destination end index.
        end: usize,
        /// Available storage capacity.
        capacity: usize,
    },
    /// A collection exceeds its supported representation limit.
    LimitExceeded {
        /// Collection or integer representation whose limit was exceeded.
        target: LimitTarget,
        /// Requested collection size.
        count: usize,
        /// Maximum supported count.
        limit: usize,
    },
    /// The active vector length exceeds its capacity.
    FeatureVectorCapacityTooSmall {
        /// Available storage capacity.
        capacity: usize,
        /// Requested number of active output cells.
        active_length: usize,
    },
    /// A definition uses an ID reserved for internal output slots.
    ReservedFeatureId {
        /// Zero-based index of the offending definition.
        definition_index: usize,
    },
    /// An indicator requires a nonempty ring buffer.
    RingBufferCapacityZero,
    /// The indicator already has its maximum number of windows.
    WindowLimitReached {
        /// Maximum supported count.
        limit: usize,
    },
    /// A window was added after the indicator began receiving data.
    WindowAddedAfterData,
    /// A window must contain at least one sample or bucket.
    WindowPeriodZero,
    /// A window exceeds the available history capacity.
    WindowPeriodExceedsCapacity {
        /// Requested window period in samples or buckets.
        period: usize,
        /// Available storage capacity.
        capacity: usize,
    },
    /// A window leaves no extra history slot required by the indicator.
    WindowPeriodMustBeLessThanCapacity {
        /// Requested window period in samples or buckets.
        period: usize,
        /// Available storage capacity.
        capacity: usize,
    },
    /// An aggregation interval is shorter than one millisecond.
    AggregationTooShort,
    /// A duration has sub-millisecond precision.
    DurationPrecision {
        /// Argument or event field that failed validation.
        field: DurationField,
    },
    /// A duration cannot fit in signed 64-bit milliseconds.
    DurationOutOfRange {
        /// Argument or event field that failed validation.
        field: DurationField,
    },
    /// A window is shorter than one aggregation interval.
    WindowShorterThanAggregation,
    /// A window is not an exact multiple of its aggregation interval.
    WindowNotMultipleOfAggregation,
    /// A period cannot fit in the required integer type.
    WindowPeriodOutOfRange {
        /// Collection or integer representation whose limit was exceeded.
        target: IntegerTarget,
    },
    /// A window duration exceeds signed 64-bit milliseconds.
    WindowDurationOutOfRange,
    /// A timed period exceeds the indicator's supported bounds.
    TimedPeriodTooLarge {
        /// Indicator family being configured.
        indicator: IndicatorKind,
    },
}

/// Collection whose fixed-size representation was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LimitTarget {
    /// Interned market symbols.
    Symbols,
    /// Compiled runtime indicator instances.
    RuntimeFeatures,
    /// Per-symbol routing entries.
    SymbolRouters,
    /// Subscribers across routing groups.
    Subscribers,
    /// Subscribers in one routing group.
    SubscriberGroup,
    /// Configured per-symbol order books.
    OrderBooks,
}

/// Duration argument involved in validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DurationField {
    /// Duration of one aggregation bucket.
    Aggregation,
    /// Total rolling-window duration.
    Window,
    /// Durations used together to configure a timed window.
    TimedWindow,
}

/// Integer representation required by an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegerTarget {
    /// A signed 64-bit integer.
    Signed64,
    /// A platform-sized unsigned integer.
    Usize,
}

/// Indicator family associated with a compiled definition or construction error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndicatorKind {
    /// Externally supplied context value.
    Context,
    /// Raw scalar event field.
    Field,
    /// Highest bid price.
    OrderBookBestBidPrice,
    /// Size at the highest bid.
    OrderBookBestBidSize,
    /// Lowest ask price.
    OrderBookBestAskPrice,
    /// Size at the lowest ask.
    OrderBookBestAskSize,
    /// Size at a specified side and price.
    OrderBookLevelSize,
    /// Price at a one-based depth.
    OrderBookNthPrice,
    /// Size at a one-based depth.
    OrderBookNthSize,
    /// Cumulative size through an inclusive price threshold.
    OrderBookDepthUntilPrice,
    /// First price used to reach a target size.
    OrderBookDepthUntilSizePriceFrom,
    /// Last price used to reach a target size.
    OrderBookDepthUntilSizePriceTo,
    /// Whole-level cumulative size used to reach a target.
    OrderBookDepthUntilSizeTotalSize,
    /// Total size in a half-open price range.
    OrderBookVolumeBetweenPrices,
    /// Midpoint of the best bid and ask.
    OrderBookMidPrice,
    /// Best ask minus best bid.
    OrderBookSpread,
    /// Spread relative to mid-price in basis points.
    OrderBookSpreadBps,
    /// Best prices weighted by their own sizes.
    OrderBookWeightedMidPrice,
    /// Best prices weighted by opposite-side sizes.
    OrderBookMicroprice,
    /// Bid-minus-ask size divided by total size.
    OrderBookImbalance,
    /// Sample-based simple moving average.
    Sma,
    /// Sample-based exponential moving average.
    Ema,
    /// Rolling aggressor buy volume minus sell volume.
    Cvd,
    /// Fractional price change over a sample lag.
    SimpleReturn,
    /// Natural logarithm of the price ratio over a sample lag.
    LogReturn,
    /// Population standard deviation of simple returns.
    Volatility,
    /// Simple moving average over time buckets.
    SmaTimed,
    /// On-balance volume over time buckets.
    ObvTimed,
    /// Population standard deviation of sample returns grouped into time buckets.
    VolatilityTimed,
    /// Cumulative volume-price trend.
    Vpt,
    /// Trade count over a rolling timed window.
    TradeCountTimed,
    /// Total trade volume over a rolling timed window.
    TradeVolumeTimed,
    /// Volume-weighted average trade price over a rolling timed window.
    VwapTimed,
    /// UTC weekday derived from an event timestamp.
    DayOfWeek,
    /// Elapsed time since the first observed event of the local day.
    TimeSinceFirstEventOfDay,
}

/// Allocation-free reason why a feature definition could not be compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidIndicatorDefinitionError {
    /// A context name must contain at least one character.
    EmptyContextName,
    /// An order-book query uses a negative price.
    OrderBookPriceNegative,
    /// An order-book target size is zero or negative.
    OrderBookSizeNotPositive,
    /// An order-book query has unordered or equal price bounds.
    OrderBookPriceRangeInvalid,
    /// An order-book depth is zero instead of one-based.
    OrderBookDepthZero,
    /// Compatible definitions exceed the outputs supported by one indicator.
    CompatibleGroupOutputLimitExceeded {
        /// Maximum supported count.
        limit: usize,
    },
    /// A scalar runtime derivation is requested more than once.
    DuplicateScalarDerivation,
    /// Two definitions describe the same structural feature.
    DuplicateFeatureKey,
    /// Two definitions use the same output ID.
    DuplicateFeatureId,
    /// The indicator requires a scalar event field as input.
    ScalarEventFieldSourceRequired,
    /// The indicator requires complete trade events as input.
    TradeEventSourceRequired,
    /// A sample window or lag is zero.
    WindowTooShort,
    /// A sample window or lag exceeds the supported maximum.
    WindowTooLarge,
    /// An aggregation interval is shorter than one millisecond.
    AggregationTooShort,
    /// A window is shorter than one aggregation interval.
    WindowShorterThanAggregation {
        /// Configured bucket duration in milliseconds.
        aggregation_millis: i64,
        /// Configured rolling duration in milliseconds.
        window_millis: i64,
    },
    /// A window is not an exact multiple of its aggregation interval.
    WindowNotMultipleOfAggregation {
        /// Configured bucket duration in milliseconds.
        aggregation_millis: i64,
        /// Configured rolling duration in milliseconds.
        window_millis: i64,
    },
    /// The number of aggregation buckets cannot fit in `usize`.
    BucketPeriodOutOfRange,
    /// A duration has sub-millisecond precision.
    DurationPrecision {
        /// Argument or event field that failed validation.
        field: DefinitionDurationField,
        /// Duration supplied by the definition.
        duration: Duration,
    },
    /// A duration cannot fit in signed 64-bit milliseconds.
    DurationOutOfRange {
        /// Argument or event field that failed validation.
        field: DefinitionDurationField,
        /// Duration supplied by the definition.
        duration: Duration,
    },
    /// A calendar offset is outside -14 to +14 hours inclusive.
    UtcOffsetOutOfRange {
        /// Requested UTC offset in milliseconds.
        offset_millis: i64,
    },
    /// A calendar offset is not an exact number of minutes.
    UtcOffsetPrecision {
        /// Requested UTC offset in milliseconds.
        offset_millis: i64,
    },
    /// An argument failed validation.
    InvalidArgument(InvalidArgumentError),
    /// Construction failed without a more specific validation reason.
    ConstructionFailed,
}

/// Allocation-free reason why a scalar model-input transformation is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidTransformationDefinitionError {
    /// Sample windows must be positive.
    WindowZero,
    /// Compatible averages exceed the calculator output limit.
    TooManyAverageOutputs,
    /// The input ID is absent from the preceding layout.
    InputFeatureNotFound,
    /// Two transformations use the same output ID.
    DuplicateOutputFeature,
    /// An output ID is reserved for internal slots.
    ReservedOutputFeature,
    /// A standard-scaler mean is NaN or infinite.
    MeanNotFinite,
    /// A standard-scaler scale is NaN or infinite.
    ScaleNotFinite,
    /// A standard-scaler scale is zero or negative.
    ScaleNotPositive,
    /// The reciprocal of the scale is not finite.
    InverseScaleNotFinite,
    /// A lag window is zero.
    LagWindowZero,
    /// A lag window exceeds 10,000 accepted events.
    LagWindowTooLarge,
}

/// Duration field in an invalid compiled feature definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DefinitionDurationField {
    /// Duration of one aggregation bucket.
    Aggregation,
    /// Total rolling-window duration.
    Window,
}

impl Display for FimlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidContextUpdate { index, reason } => {
                write!(f, "context update {index}: {reason}")
            }
            FimlError::InvalidPipelineStage { index, reason } => {
                write!(f, "invalid pipeline stage {index}: {reason}")
            }
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
            FimlError::InvalidTransformationDefinition { index, reason } => {
                write!(
                    f,
                    "invalid transformation definition at index {index}: {reason}"
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
            FimlError::ModelVectorLengthMismatch { expected, actual } => write!(
                f,
                "model-input storage has {actual} active cells, but the model-input spec requires exactly {expected}"
            ),
            FimlError::ModelVectorCapacityMismatch { expected, actual } => write!(
                f,
                "model-input storage has capacity {actual}, but the model-input spec requires capacity {expected}"
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
            Self::GlobalOrderBookSymbol => f.write_str("order books require a non-global symbol"),
            Self::NonFiniteEventValue { field } => {
                let name = match field {
                    EventField::Price => "price",
                    EventField::Volume => "volume",
                    EventField::TradePrice => "trade price",
                    EventField::TradeVolume => "trade volume",
                };
                write!(f, "event {name} must be finite")
            }
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
            Self::Symbols => "symbol",
            Self::RuntimeFeatures => "runtime feature",
            Self::SymbolRouters => "symbol router",
            Self::Subscribers => "subscriber",
            Self::SubscriberGroup => "subscriber group",
            Self::OrderBooks => "order-book",
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
            Self::OrderBookBestBidPrice => "order-book best bid price",
            Self::OrderBookBestBidSize => "order-book best bid size",
            Self::OrderBookBestAskPrice => "order-book best ask price",
            Self::OrderBookBestAskSize => "order-book best ask size",
            Self::OrderBookLevelSize => "order book level size",
            Self::OrderBookNthPrice => "order book nth price",
            Self::OrderBookNthSize => "order book nth size",
            Self::OrderBookDepthUntilPrice => "order book depth until price",
            Self::OrderBookDepthUntilSizePriceFrom => "order book depth until size price from",
            Self::OrderBookDepthUntilSizePriceTo => "order book depth until size price to",
            Self::OrderBookDepthUntilSizeTotalSize => "order book depth until size total size",
            Self::OrderBookVolumeBetweenPrices => "order book volume between prices",
            Self::OrderBookMidPrice => "order-book mid-price",
            Self::OrderBookSpread => "order-book spread",
            Self::OrderBookSpreadBps => "order-book spread in basis points",
            Self::OrderBookWeightedMidPrice => "order-book weighted mid-price",
            Self::OrderBookMicroprice => "order-book microprice",
            Self::OrderBookImbalance => "order-book imbalance",
            Self::Context => "context",
            Self::Field => "field",
            Self::Sma => "SMA",
            Self::Ema => "EMA",
            Self::Cvd => "CVD",
            Self::SimpleReturn => "simple return",
            Self::LogReturn => "log return",
            Self::Volatility => "rolling volatility",
            Self::SmaTimed => "timed SMA",
            Self::ObvTimed => "timed OBV",
            Self::VolatilityTimed => "timed rolling volatility",
            Self::Vpt => "VPT",
            Self::TradeCountTimed => "timed trade count",
            Self::TradeVolumeTimed => "timed trade volume",
            Self::VwapTimed => "timed VWAP",
            Self::DayOfWeek => "day of week",
            Self::TimeSinceFirstEventOfDay => "time since first event of day",
        })
    }
}

impl Display for InvalidIndicatorDefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrderBookPriceNegative => {
                f.write_str("order-book query prices must be nonnegative")
            }
            Self::OrderBookSizeNotPositive => {
                f.write_str("order-book target size must be positive")
            }
            Self::OrderBookPriceRangeInvalid => {
                f.write_str("order-book query requires from_price < to_price")
            }
            Self::OrderBookDepthZero => f.write_str("order-book depth must be positive"),
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
            Self::EmptyContextName => f.write_str("context name must not be empty"),
            Self::ConstructionFailed => f.write_str("indicator construction failed"),
        }
    }
}

impl Display for InvalidTransformationDefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InputFeatureNotFound => "input feature ID does not exist in the preceding layout",
            Self::DuplicateOutputFeature => "output feature ID duplicates an earlier output",
            Self::ReservedOutputFeature => "output feature ID uses the reserved namespace",
            Self::MeanNotFinite => "standard-scaler mean must be finite",
            Self::ScaleNotFinite => "standard-scaler scale must be finite",
            Self::ScaleNotPositive => "standard-scaler scale must be positive",
            Self::InverseScaleNotFinite => "standard-scaler inverse scale must be finite",
            Self::WindowZero => "average window must be positive",
            Self::TooManyAverageOutputs => "too many compatible average outputs (maximum 16)",
            Self::LagWindowZero => "lag window must be positive",
            Self::LagWindowTooLarge => "lag window exceeds the maximum of 10000 values",
        })
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

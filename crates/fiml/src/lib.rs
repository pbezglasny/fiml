pub mod event;
pub mod features;
pub mod indicators;
pub mod order_book;
mod ring_buffer;
pub mod symbols;
mod types;
mod vectors;

use std::{error::Error, fmt::Display};

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

pub type Result<T> = std::result::Result<T, FimlError>;

#[derive(Debug)]
#[non_exhaustive]
pub enum FimlError {
    InvalidArgument(String),
    InvalidPriceRange {
        from_price: Decimal,
        to_price: Decimal,
    },
    InvalidIndicatorDefinition {
        index: usize,
        reason: String,
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
}

impl Display for FimlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FimlError::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
            FimlError::InvalidPriceRange {
                from_price,
                to_price,
            } => write!(
                f,
                "invalid price range: from price {from_price} must be less than to price {to_price}"
            ),
            FimlError::InvalidIndicatorDefinition { index, reason } => {
                write!(f, "invalid indicator definition at index {index}: {reason}")
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
        }
    }
}

impl Error for FimlError {}

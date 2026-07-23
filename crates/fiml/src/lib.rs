pub mod features;
pub mod indicators;
mod ring_buffer;
pub mod symbols;
mod types;
mod vectors;

use std::{error::Error, fmt::Display};

#[cfg(feature = "serde")]
pub use features::FEATURE_SET_FORMAT_VERSION;
pub use features::{
    DispatchSequenceError, Event, EventKind, Feature, FeatureExtractor, FeatureSet,
    FeatureSetBuilder, IndicatorDef, IndicatorFeatureVector, IndicatorFeatures, IndicatorSpec,
    MAX_OUTPUTS_PER_INDICATOR, OrderBookUpdate, PriceUpdate, TimeUpdate, TimeWindows, TradeSide,
    TradeUpdate, ValueSource, VolumeUpdate,
};
pub use indicators::{CumulativeVolumeDelta, ObvBucket, OnBalanceVolumeTimed};
pub use ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
pub use symbols::Symbol;
pub use types::Float;
pub use vectors::{ArrayFeatureVector, FeatureVector, VecFeatureVector};

pub type Result<T> = std::result::Result<T, FimlError>;

#[derive(Debug)]
#[non_exhaustive]
pub enum FimlError {
    InvalidArgument(String),
    InvalidIndicatorDefinition {
        index: usize,
        reason: String,
    },
    TooManyIndicators {
        count: usize,
        capacity: usize,
    },
    TooManyOutputs {
        count: usize,
        capacity: usize,
    },
    OutputCountMismatch {
        expected: usize,
        actual: usize,
    },
    TimestampOutOfOrder {
        symbol: Option<Symbol>,
        event_kind: EventKind,
        timestamp: i64,
        previous_timestamp: i64,
    },
}

impl Display for FimlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FimlError::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
            FimlError::InvalidIndicatorDefinition { index, reason } => {
                write!(f, "invalid indicator definition at index {index}: {reason}")
            }
            FimlError::TooManyIndicators { count, capacity } => {
                write!(
                    f,
                    "indicator count {count} exceeds fixed capacity {capacity}"
                )
            }
            FimlError::TooManyOutputs { count, capacity } => {
                write!(f, "output count {count} exceeds fixed capacity {capacity}")
            }
            FimlError::OutputCountMismatch { expected, actual } => {
                write!(
                    f,
                    "output storage has {actual} cells, but compilation requires exactly {expected}"
                )
            }
            FimlError::TimestampOutOfOrder {
                symbol,
                event_kind,
                timestamp,
                previous_timestamp,
            } => {
                write!(f, "timestamp {timestamp} for {event_kind}")?;
                if let Some(symbol) = symbol {
                    write!(
                        f,
                        " event for symbol {}",
                        symbols::resolve(*symbol).unwrap_or_else(|| format!("{symbol:?}"))
                    )?;
                }
                write!(
                    f,
                    " is earlier than previous timestamp {previous_timestamp}"
                )
            }
        }
    }
}

impl Error for FimlError {}

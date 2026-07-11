pub mod features;
pub mod indicators;
mod ring_buffer;
pub mod symbols;
mod types;
mod vectors;

use std::{error::Error, fmt::Display};

pub use features::{
    Event, EventKind, Feature, FeatureDef, FeatureExtractor, FeatureSet, IndicatorFeatureVector,
    IndicatorFeatures, IndicatorSpec, OrderBookUpdate, PriceUpdate, TimeUnit, TimeUpdate,
    TradeUpdate, VolumeUpdate,
};
pub use indicators::{
    IndicatorFeatureVectorBuilder, ObvBucket, ObvTimedPeriodsBuilder, OnBalanceVolumeTimed,
};
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

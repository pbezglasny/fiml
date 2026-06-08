pub mod builder;
pub mod features;
#[allow(dead_code)]
pub mod indicators;
mod ring_buffer;
pub mod ticker;
mod types;
mod vectors;

use std::{error::Error, fmt::Display};

pub use builder::IndicatorFeatureVectorBuilder;
pub use features::{
    Event, EventKind, Feature, IndicatorFeatureVector, OrderBookUpdate, PriceUpdate, TimeUpdate,
};
pub use ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
pub use ticker::Ticker;
pub use types::Float;
pub use vectors::{ArrayFeatureVector, FeatureOutput, FeatureVector};

pub type Result<T> = std::result::Result<T, FimlError>;

#[derive(Debug)]
#[non_exhaustive]
pub enum FimlError {
    InvalidArgument(String),
}

impl Display for FimlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FimlError::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
        }
    }
}

impl Error for FimlError {}

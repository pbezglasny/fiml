pub mod features;
#[allow(dead_code)]
mod indicators;
mod ring_buffer;
mod types;
mod vectors;

use std::{error::Error, fmt::Display};

pub use features::{Event, EventKind, Feature, OrderBookUpdate, PriceUpdate, TimeUpdate};
pub use indicators::averages::SimpleMovingAverage;
pub use ring_buffer::{
    HeapRingBuffer, RingBuffer, StackRingBuffer, new_heap_ring_buffer, new_stack_ring_buffer,
};
pub use types::Float;
pub use vectors::{ArrayFeatureVector, FeatureOutput, FeatureVector, Handler};

pub type Result<T> = std::result::Result<T, FimlError>;

#[derive(Debug)]
#[non_exhaustive]
pub enum FimlError {
    InvalidArgument(String),
    FeatureVectorFull,
}

impl Display for FimlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FimlError::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
            FimlError::FeatureVectorFull => write!(f, "feature vector is full"),
        }
    }
}

impl Error for FimlError {}

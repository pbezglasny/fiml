#[allow(dead_code)]
mod indicators;
mod ring_buffer;
mod types;

use std::{error::Error, fmt::Display};

pub use indicators::averages::SimpleMovingAverage;
pub use ring_buffer::{HeapRingBuffer, RingBuffer, StackRingBuffer};
pub use types::Float;

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

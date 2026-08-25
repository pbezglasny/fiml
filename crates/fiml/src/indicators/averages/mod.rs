mod ema;
mod sma;

pub use ema::{ExponentialMovingAverage, ema};
pub use sma::{SimpleMovingAverage, SimpleMovingAverageTimed, sma, sma_timed};

mod ema;
mod sma;

pub(crate) use ema::PendingEmaPeriods;
pub use ema::{EmaPeriodsBuilder, ExponentialMovingAverage};
pub(crate) use sma::{PendingSmaPeriods, PendingSmaTimedPeriods};
pub use sma::{
    SimpleMovingAverage, SimpleMovingAverageTimed, SmaPeriodsBuilder, SmaTimedPeriodsBuilder,
};

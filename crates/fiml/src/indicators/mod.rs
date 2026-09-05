//! Standalone streaming indicators and their window state.
//!
//! Direct callers must supply finite numeric inputs; indicator updates do not
//! validate NaN or infinity and can retain poisoned state. Event ingestion via
//! [`crate::FeatureExtractor`] or [`crate::Pipeline`] rejects these inputs.

pub mod averages;
pub mod counts;
pub mod volume;

pub use averages::{
    ExponentialMovingAverage, SimpleMovingAverage, SimpleMovingAverageTimed, ema, sma, sma_timed,
};
pub use counts::{CountBucket, TradeCountTimed, trade_count_timed};
pub use volume::{CumulativeVolumeDelta, ObvBucket, OnBalanceVolumeTimed, cvd, obv_timed};

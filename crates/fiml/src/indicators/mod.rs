//! Standalone streaming indicators and their window state.
//!
//! Direct callers must supply finite numeric inputs; indicator updates do not
//! validate NaN or infinity and can retain poisoned state. Event ingestion via
//! [`crate::FeatureExtractor`] or [`crate::Pipeline`] rejects these inputs.
//!
//! Timed indicators take explicit epoch-millisecond timestamps in `update`.
//! Call `observe` to advance expiry and warm-up without adding data; reads do
//! not advance time. Timestamps must be nondecreasing across both operations.

pub mod averages;
pub mod counts;
pub mod returns;
mod timed_sum;
pub mod volatility;
pub mod volume;

pub use averages::{
    ExponentialMovingAverage, SimpleMovingAverage, SimpleMovingAverageTimed, ema, sma, sma_timed,
};
pub use counts::{CountBucket, TradeCountTimed, trade_count_timed};
pub use returns::{ReturnKind, SampleReturns};
pub use volatility::{RollingVolatility, RollingVolatilityTimed, VolatilityBucket};
pub use volume::{
    CumulativeVolumeDelta, ObvBucket, OnBalanceVolumeTimed, RollingTradeVolumeTimed,
    TradeVolumeBucket, VolumePriceTrend, VolumeWeightedAveragePriceTimed, VwapBucket, cvd,
    obv_timed, trade_volume_timed, vpt, vwap_timed,
};

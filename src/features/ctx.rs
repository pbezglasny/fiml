use crate::Float;

/// Input bundle passed to every feature on each tick.
///
/// A feature reads only the fields it needs and ignores the rest. New input
/// streams (bid/ask/volume/...) are added by extending this struct, so the
/// [`Feature`](crate::features::Feature) trait signature stays stable.
pub struct UpdateCtx<F: Float> {
    /// Price value for this tick.
    pub value: F,
    /// Unix timestamp in seconds.
    pub timestamp: i64,
}

impl<F: Float> UpdateCtx<F> {
    pub fn new(value: F, timestamp: i64) -> Self {
        Self { value, timestamp }
    }
}

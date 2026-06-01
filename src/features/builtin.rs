use crate::features::ctx::UpdateCtx;
use crate::features::feature::Feature;
use crate::{Float, Handler, HeapRingBuffer, SimpleMovingAverage};

/// Maximum number of SMA windows that can share a single indicator instance.
/// Exceeding it during construction is an error.
pub const MAX_WINDOWS_PER_SMA: usize = 16;

/// Day-of-week feature. Writes `0 = Sunday ..= 6 = Saturday` derived from the
/// tick timestamp through its handler. A non-price builtin: it reads
/// `ctx.timestamp` and ignores `ctx.value`.
pub struct DayOfWeek<'a, F: Float> {
    handler: Handler<'a, F>,
}

impl<'a, F: Float> DayOfWeek<'a, F> {
    pub fn new(handler: Handler<'a, F>) -> Self {
        Self { handler }
    }
}

impl<F: Float> Feature<F> for DayOfWeek<'_, F> {
    fn update(&mut self, ctx: &UpdateCtx<F>) {
        // Unix epoch (1970-01-01) was a Thursday, index 4 in a Sunday-based week.
        let days = ctx.timestamp.div_euclid(86_400);
        let dow = (days + 4).rem_euclid(7);
        self.handler.set_value(F::from_usize(dow as usize));
    }
}

/// Closed enum of features shipped by the library.
///
/// Dispatched statically: each [`update`](Feature::update) is a `match` of
/// direct calls, no `Box` and no vtable. Users needing custom features wrap
/// this in their own enum (see the module docs).
pub enum BuiltinFeature<'a, F: Float> {
    Sma(SimpleMovingAverage<'a, HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>),
    DayOfWeek(DayOfWeek<'a, F>),
    // future: Ema(...), Rsi(...), ...
}

impl<F: Float> Feature<F> for BuiltinFeature<'_, F> {
    fn update(&mut self, ctx: &UpdateCtx<F>) {
        match self {
            BuiltinFeature::Sma(s) => s.update(ctx.value),
            BuiltinFeature::DayOfWeek(d) => d.update(ctx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn sma_dispatches_through_feature_trait() {
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let handler = fv.next_handler();
        let mut sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, MAX_WINDOWS_PER_SMA> =
            SimpleMovingAverage::new_heap(3);
        sma.add_window_with_handler(3, handler).unwrap();

        let mut feat = BuiltinFeature::Sma(sma);
        for v in [3.0, 6.0, 9.0] {
            feat.update(&UpdateCtx::new(v, 0));
        }
        drop(feat);

        // (3 + 6 + 9) / 3 = 6.0
        assert!(approx_eq(fv.values()[0], 6.0));
    }

    #[test]
    fn day_of_week_writes_weekday() {
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let handler = fv.next_handler();
        let mut feat = BuiltinFeature::DayOfWeek(DayOfWeek::new(handler));

        // 2021-01-01 00:00:00 UTC was a Friday (index 5, Sunday-based).
        feat.update(&UpdateCtx::new(0.0, 1_609_459_200));
        drop(feat);

        assert!(approx_eq(fv.values()[0], 5.0));
    }
}

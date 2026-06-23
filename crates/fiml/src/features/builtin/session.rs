use crate::features::BuiltinFeature;
use crate::features::event::{Event, FeatureRoute};
use crate::features::indicator_vector::{BuiltinFeatureEntry, FeatureKey};
use crate::vectors::FeatureVector;
use crate::{Float, Symbol};

/// Milliseconds in a day. Event timestamps are epoch milliseconds, so the
/// calendar day index is the (timezone-shifted) timestamp divided by this.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// Time elapsed since the trading session opened, in **milliseconds**.
///
/// An every-event clock feature: it refreshes from each event's timestamp
/// regardless of kind. The session open is *inferred from the stream* — the
/// first event seen after a day boundary — rather than hard-coded exchange hours,
/// so the Python (batch) and Rust (live) sides derive the same boundary from the
/// same events. The day boundary is taken in a fixed UTC offset (`utc_offset_millis`,
/// `0` = UTC), the feature's only knob.
pub struct TimeSinceSessionOpen {
    output_index: usize,
    /// Timezone offset applied before computing the day boundary, in milliseconds.
    utc_offset_millis: i64,
    /// Day index (in the offset timezone) of the current session, or `None`
    /// before the first event.
    current_day: Option<i64>,
    /// Raw timestamp of the first event seen in the current session.
    session_open_ts: i64,
}

impl TimeSinceSessionOpen {
    pub fn new(output_index: usize, utc_offset_millis: i64) -> Self {
        Self {
            output_index,
            utc_offset_millis,
            current_day: None,
            session_open_ts: 0,
        }
    }

    pub fn update<F: Float, O: FeatureVector<F = F>>(&mut self, timestamp: i64, output: &mut O) {
        let day = (timestamp + self.utc_offset_millis).div_euclid(MILLIS_PER_DAY);
        if self.current_day != Some(day) {
            self.current_day = Some(day);
            self.session_open_ts = timestamp;
        }
        // Non-negative under in-order replay (session open is the first event of
        // the day); clamp defensively so out-of-order ticks never wrap.
        let elapsed = (timestamp - self.session_open_ts).max(0);
        output.set_value_at(self.output_index, F::from_usize(elapsed as usize));
    }

    pub(in crate::features) fn update_event<F: Float, O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        self.update(event.timestamp(), output);
    }
}

pub(in crate::features) fn build_builtin<F: Float>(
    utc_offset_millis: i64,
    output_index: usize,
) -> BuiltinFeature<F> {
    BuiltinFeature::TimeSinceSessionOpen(TimeSinceSessionOpen::new(output_index, utc_offset_millis))
}

pub(crate) fn build_entry<F: Float>(
    symbol: Symbol,
    utc_offset_millis: i64,
    output_index: usize,
    names: &mut [Option<FeatureKey>],
) -> BuiltinFeatureEntry<F> {
    names[output_index] = Some(FeatureKey {
        symbol,
        name: "time_since_session_open".to_string(),
    });
    BuiltinFeatureEntry {
        feature: build_builtin(utc_offset_millis, output_index),
        route: FeatureRoute::Every,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn measures_elapsed_since_first_event_of_the_day() {
        let aapl = symbols::intern("AAPL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut feat = TimeSinceSessionOpen::new(0, 0);

        // First event of the day opens the session: elapsed is zero.
        let open = 1_609_459_200_000; // 2021-01-01 00:00:00 UTC
        feat.update_event(&Event::price(aapl, 10.0, open), &mut fv);
        assert!(approx_eq(fv.values()[0], 0.0));

        // Later same-day event: elapsed grows from the session open.
        feat.update_event(&Event::trade(aapl, 11.0, 1.0, open + 5_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 5_000.0));
    }

    #[test]
    fn resets_at_the_next_day_boundary() {
        let aapl = symbols::intern("AAPL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut feat = TimeSinceSessionOpen::new(0, 0);

        let open = 1_609_459_200_000; // 2021-01-01 00:00:00 UTC
        feat.update_event(&Event::price(aapl, 10.0, open + 3_600_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 0.0)); // first event of day 1

        // First event of the next day re-opens the session.
        let next_day = open + MILLIS_PER_DAY + 2_000;
        feat.update_event(&Event::price(aapl, 12.0, next_day), &mut fv);
        assert!(approx_eq(fv.values()[0], 0.0));
        feat.update_event(&Event::price(aapl, 13.0, next_day + 1_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 1_000.0));
    }
}

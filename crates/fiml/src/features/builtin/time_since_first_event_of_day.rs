use crate::Float;
use crate::features::BuiltinFeature;
use crate::features::compiler::OutputSpan;
use crate::features::event::Event;
use crate::vectors::FeatureVector;

/// Milliseconds in a day. Event timestamps are epoch milliseconds, so the
/// calendar day index is the (timezone-shifted) timestamp divided by this.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// Time elapsed since the first observed event of the local day, in milliseconds.
///
/// An every-event clock feature: it refreshes from each event's timestamp
/// regardless of kind. The day boundary uses a fixed UTC offset
/// (`utc_offset_millis`, `0` = UTC).
pub struct TimeSinceFirstEventOfDay {
    output_span: OutputSpan,
    /// Timezone offset applied before computing the day boundary, in milliseconds.
    utc_offset_millis: i64,
    /// Day index (in the offset timezone) of the current day, or `None`
    /// before the first event.
    current_day: Option<i128>,
    /// Raw timestamp of the first event seen in the current day.
    first_event_timestamp: i64,
}

impl TimeSinceFirstEventOfDay {
    pub(crate) fn new(output_span: OutputSpan, utc_offset_millis: i64) -> Self {
        debug_assert_eq!(output_span.count, 1);
        Self {
            output_span,
            utc_offset_millis,
            current_day: None,
            first_event_timestamp: 0,
        }
    }

    pub fn update<F: Float, O: FeatureVector<F = F>>(&mut self, timestamp: i64, output: &mut O) {
        let day = (i128::from(timestamp) + i128::from(self.utc_offset_millis))
            .div_euclid(i128::from(MILLIS_PER_DAY));
        if self.current_day != Some(day) {
            self.current_day = Some(day);
            self.first_event_timestamp = timestamp;
        }
        let elapsed = timestamp.saturating_sub(self.first_event_timestamp).max(0);
        output.set_value_at(self.output_span.start, F::from_usize(elapsed as usize));
    }

    pub(in crate::features) fn update_event<F: Float, O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        self.update(event.timestamp(), output);
    }
}

pub(crate) fn build<F: Float>(
    utc_offset_millis: i64,
    output_span: OutputSpan,
) -> BuiltinFeature<F> {
    BuiltinFeature::TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDay::new(
        output_span,
        utc_offset_millis,
    ))
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
        let mut feat = TimeSinceFirstEventOfDay::new(OutputSpan { start: 0, count: 1 }, 0);

        // First event of the day establishes the origin: elapsed is zero.
        let open = 1_609_459_200_000; // 2021-01-01 00:00:00 UTC
        feat.update_event(&Event::price(aapl, 10.0, open), &mut fv);
        assert!(approx_eq(fv.values()[0], 0.0));

        // Later same-day event: elapsed grows from the first event.
        feat.update_event(&Event::trade(aapl, 11.0, 1.0, open + 5_000, None), &mut fv);
        assert!(approx_eq(fv.values()[0], 5_000.0));
    }

    #[test]
    fn resets_at_the_next_day_boundary() {
        let aapl = symbols::intern("AAPL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut feat = TimeSinceFirstEventOfDay::new(OutputSpan { start: 0, count: 1 }, 0);

        let open = 1_609_459_200_000; // 2021-01-01 00:00:00 UTC
        feat.update_event(&Event::price(aapl, 10.0, open + 3_600_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 0.0)); // first event of day 1

        // First event of the next day establishes a new origin.
        let next_day = open + MILLIS_PER_DAY + 2_000;
        feat.update_event(&Event::price(aapl, 12.0, next_day), &mut fv);
        assert!(approx_eq(fv.values()[0], 0.0));
        feat.update_event(&Event::price(aapl, 13.0, next_day + 1_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 1_000.0));
    }
}

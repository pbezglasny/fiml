use crate::Float;
use crate::features::BuiltinFeature;
use crate::features::compiler::OutputSpan;
use crate::features::event::Event;
use crate::vectors::FeatureVector;

/// Milliseconds in a day. Event timestamps are epoch milliseconds, so the
/// calendar day index is the timestamp divided by this.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// Day-of-week feature. Writes `0 = Sunday ..= 6 = Saturday` derived from the
/// event timestamp to its output cell. An every-event clock feature: it refreshes
/// from each event's timestamp regardless of kind, so it has a value on every row.
pub struct DayOfWeek {
    output_span: OutputSpan,
}

impl DayOfWeek {
    pub(crate) fn new(output_span: OutputSpan) -> Self {
        debug_assert_eq!(output_span.count, 1);
        Self { output_span }
    }

    pub fn update<F: Float, O: FeatureVector<F = F>>(&mut self, timestamp: i64, output: &mut O) {
        // Unix epoch (1970-01-01) was a Thursday, index 4 in a Sunday-based week.
        let days = timestamp.div_euclid(MILLIS_PER_DAY);
        let dow = (days + 4).rem_euclid(7);
        output.set_value_at(self.output_span.start, F::from_usize(dow as usize));
    }

    pub(in crate::features) fn update_event<F: Float, O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        self.update(event.timestamp(), output);
    }
}

pub(crate) fn build<F: Float>(output_span: OutputSpan) -> BuiltinFeature<F> {
    BuiltinFeature::DayOfWeek(DayOfWeek::new(output_span))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn day_of_week_reacts_to_every_event() {
        let aapl = symbols::intern("AAPL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut feat = DayOfWeek::new(OutputSpan { start: 0, count: 1 });

        // A price event carries a timestamp too, so the clock feature updates from
        // it without needing an explicit time event. 2021-01-01 was a Friday (5),
        // in epoch milliseconds.
        feat.update_event(&Event::price(aapl, 42.0, 1_609_459_200_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 5.0));

        // 2021-01-02 (Saturday, 6) one day later, via a time event.
        feat.update_event(&Event::time(1_609_545_600_000), &mut fv);
        assert!(approx_eq(fv.values()[0], 6.0));
    }
}

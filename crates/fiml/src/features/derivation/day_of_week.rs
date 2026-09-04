use crate::event::Event;
use crate::features::compiler::OutputSpan;
use crate::features::derivation::FeatureDerivation;
use crate::vectors::FeatureVector;

/// Milliseconds in a day. Event timestamps are epoch milliseconds, so the
/// calendar day index is the timestamp divided by this.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// Day-of-week feature. Writes `0 = Sunday ..= 6 = Saturday` derived from the
/// event timestamp to its output cell. An any-event clock feature: it refreshes
/// from each event's timestamp regardless of kind, so it has a value on every row.
pub(crate) struct DayOfWeek;

impl DayOfWeek {
    pub(in crate::features) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_span: OutputSpan,
        output: &mut O,
    ) {
        // Unix epoch (1970-01-01) was a Thursday, index 4 in a Sunday-based week.
        let days = event.timestamp().div_euclid(MILLIS_PER_DAY);
        let dow = (days + 4).rem_euclid(7);
        output.set_value_at(output_span.start, dow as f64);
    }
}

pub(crate) fn build() -> FeatureDerivation {
    FeatureDerivation::DayOfWeek(DayOfWeek)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn day_of_week_reacts_to_any_event() {
        let aapl = symbols::intern("AAPL");
        let mut fv: ArrayFeatureVector<1> = ArrayFeatureVector::new();
        let mut feat = DayOfWeek;
        let output_span = OutputSpan { start: 0, count: 1 };

        // A price event carries a timestamp too, so the clock feature updates from
        // it without needing an explicit time event. 2021-01-01 was a Friday (5),
        // in epoch milliseconds.
        feat.update(
            &Event::price(aapl, 42.0, 1_609_459_200_000),
            output_span,
            &mut fv,
        );
        assert!(approx_eq(fv.values()[0], 5.0));

        // 2021-01-02 (Saturday, 6) one day later, via a time event.
        feat.update(&Event::time(1_609_545_600_000), output_span, &mut fv);
        assert!(approx_eq(fv.values()[0], 6.0));
    }
}

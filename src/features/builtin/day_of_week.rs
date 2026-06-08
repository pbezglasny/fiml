use crate::features::BuiltinFeature;
use crate::features::event::{Event, EventKind, TimeUpdate};
use crate::features::vector::{BuiltinFeatureEntry, FeatureKey};
use crate::vectors::FeatureOutput;
use crate::{Float, Ticker};

/// Day-of-week feature. Writes `0 = Sunday ..= 6 = Saturday` derived from the
/// tick timestamp to its output cell. A non-price builtin: it reacts to
/// [`TimeUpdate`] events only.
pub struct DayOfWeek {
    output_index: usize,
}

impl DayOfWeek {
    pub fn new(output_index: usize) -> Self {
        Self { output_index }
    }

    pub fn update<F: Float, O: FeatureOutput<F>>(&mut self, ev: &TimeUpdate, output: &mut O) {
        // Unix epoch (1970-01-01) was a Thursday, index 4 in a Sunday-based week.
        let days = ev.timestamp.div_euclid(86_400);
        let dow = (days + 4).rem_euclid(7);
        output.set_value_at(self.output_index, F::from_usize(dow as usize));
    }

    pub(in crate::features) fn update_event<F: Float, O: FeatureOutput<F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Event::Time(t) = event {
            self.update(t, output);
        }
    }
}

pub(crate) fn build_entry<F: Float>(
    ticker: Ticker,
    output_index: usize,
    names: &mut [Option<FeatureKey>],
) -> BuiltinFeatureEntry<F> {
    names[output_index] = Some(FeatureKey {
        ticker,
        name: "day_of_week".to_string(),
    });
    BuiltinFeatureEntry {
        feature: BuiltinFeature::DayOfWeek(DayOfWeek::new(output_index)),
        kind: EventKind::Time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, ticker};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn day_of_week_reacts_to_time_events() {
        let aapl = ticker::intern("AAPL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut feat = DayOfWeek::new(0);

        feat.update_event(&Event::price(aapl, 42.0, 0), &mut fv);
        feat.update_event(&Event::time(1_609_459_200), &mut fv);

        assert!(approx_eq(fv.values()[0], 5.0));
    }
}

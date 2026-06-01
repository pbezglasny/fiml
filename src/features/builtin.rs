use crate::features::event::{Event, TimeUpdate};
use crate::features::feature::Feature;
use crate::vectors::FeatureOutput;
use crate::{Float, HeapRingBuffer, SimpleMovingAverage};

/// Maximum number of SMA windows that can share a single indicator instance.
/// Exceeding it during construction is an error.
pub const MAX_WINDOWS_PER_SMA: usize = 16;

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
}

/// Closed enum of features shipped by the library.
///
/// Dispatched statically: each [`update`](Feature::update) is a `match` of
/// direct calls, no `Box` and no vtable. Users needing custom features wrap
/// this in their own enum (see the module docs).
pub enum BuiltinFeature<F: Float + 'static> {
    Sma {
        sma: SimpleMovingAverage<'static, HeapRingBuffer<F>, F, MAX_WINDOWS_PER_SMA>,
        output_index: usize,
    },
    DayOfWeek(DayOfWeek),
}

impl<F: Float + 'static> Feature<F> for BuiltinFeature<F> {
    fn update<O: FeatureOutput<F>>(&mut self, event: &Event<F>, output: &mut O) {
        match self {
            BuiltinFeature::Sma { sma, output_index } => {
                if let Event::Price(p) = event {
                    sma.update(p.value);
                    if let Some(value) = sma.value_at(0) {
                        output.set_value_at(*output_index, value);
                    }
                }
            }
            BuiltinFeature::DayOfWeek(d) => {
                if let Event::Time(t) = event {
                    d.update(t, output);
                }
            }
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
    fn sma_reacts_to_price_events() {
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut sma: SimpleMovingAverage<HeapRingBuffer<f64>, f64, MAX_WINDOWS_PER_SMA> =
            SimpleMovingAverage::new_heap(3);
        sma.add_window(3).unwrap();

        let mut feat = BuiltinFeature::Sma {
            sma,
            output_index: 0,
        };
        for v in [3.0, 6.0, 9.0] {
            feat.update(&Event::price(v, 0), &mut fv);
        }
        // A non-price event is ignored.
        feat.update(&Event::time(123), &mut fv);

        // (3 + 6 + 9) / 3 = 6.0
        assert!(approx_eq(fv.values()[0], 6.0));
    }

    #[test]
    fn day_of_week_reacts_to_time_events() {
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut feat = BuiltinFeature::<f64>::DayOfWeek(DayOfWeek::new(0));

        // A price event is ignored; only the time event sets the value.
        feat.update(&Event::price(42.0, 0), &mut fv);
        // 2021-01-01 00:00:00 UTC was a Friday (index 5, Sunday-based).
        feat.update(&Event::time(1_609_459_200), &mut fv);

        assert!(approx_eq(fv.values()[0], 5.0));
    }
}

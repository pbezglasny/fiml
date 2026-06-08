use crate::Float;
use crate::features::event::Event;
use crate::features::feature::Feature;
use crate::vectors::FeatureVector;

pub(crate) mod day_of_week;
pub(crate) mod ema;
pub(crate) mod sma;

pub use crate::builder::{EmaPeriodsBuilder, SmaPeriodsBuilder, SmaTimedPeriodsBuilder};
pub use day_of_week::DayOfWeek;
pub use ema::{EmaFeature, MAX_WINDOWS_PER_EMA};
pub use sma::{MAX_WINDOWS_PER_SMA, SmaFeature, SmaTimedFeature};

/// Closed enum of features shipped by the library.
///
/// Dispatched statically: each [`update`](Feature::update) is a `match` of
/// direct calls, no `Box` and no vtable. Users needing custom features wrap
/// this in their own enum (see the module docs).
pub enum BuiltinFeature<F: Float> {
    Sma(SmaFeature<F>),
    Ema(EmaFeature<F>),
    SmaTimed(SmaTimedFeature<F>),
    DayOfWeek(DayOfWeek),
}

impl<F: Float> Feature<F> for BuiltinFeature<F> {
    fn update<O: FeatureVector<F>>(&mut self, event: &Event<F>, output: &mut O) {
        match self {
            BuiltinFeature::Sma(sma) => sma.update(event, output),
            BuiltinFeature::Ema(ema) => ema.update(event, output),
            BuiltinFeature::SmaTimed(sma) => sma.update(event, output),
            BuiltinFeature::DayOfWeek(day_of_week) => day_of_week.update_event(event, output),
        }
    }
}

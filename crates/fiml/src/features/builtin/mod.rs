use crate::Float;
use crate::features::event::Event;
use crate::features::indicator_vector::Feature;
use crate::vectors::FeatureVector;

pub(crate) mod day_of_week;
pub(crate) mod ema;
pub(crate) mod obv;
pub(crate) mod session;
pub(crate) mod sma;
pub(crate) mod trade_count;

pub use crate::indicators::{
    EmaPeriodsBuilder, ObvTimedPeriodsBuilder, SmaPeriodsBuilder, SmaTimedPeriodsBuilder,
};
pub use day_of_week::DayOfWeek;
pub use ema::{EmaFeature, MAX_WINDOWS_PER_EMA};
pub use obv::{MAX_WINDOWS_PER_OBV, ObvTimedFeature};
pub use session::TimeSinceSessionOpen;
pub use sma::{MAX_WINDOWS_PER_SMA, SmaFeature, SmaTimedFeature};
pub use trade_count::TradeCountTimedFeature;

/// Closed enum of features shipped by the library.
///
/// Dispatched statically: each [`update`](Feature::update) is a `match` of
/// direct calls, no `Box` and no vtable. Users needing custom features wrap
/// this in their own enum (see the module docs).
pub enum BuiltinFeature<F: Float> {
    Sma(SmaFeature<F>),
    Ema(EmaFeature<F>),
    SmaTimed(SmaTimedFeature<F>),
    ObvTimed(ObvTimedFeature<F>),
    TradeCountTimed(TradeCountTimedFeature<F>),
    DayOfWeek(DayOfWeek),
    TimeSinceSessionOpen(TimeSinceSessionOpen),
}

impl<F: Float> Feature<F> for BuiltinFeature<F> {
    fn update<O: FeatureVector<F = F>>(&mut self, event: &Event<F>, output: &mut O) {
        match self {
            BuiltinFeature::Sma(sma) => sma.update(event, output),
            BuiltinFeature::Ema(ema) => ema.update(event, output),
            BuiltinFeature::SmaTimed(sma) => sma.update(event, output),
            BuiltinFeature::ObvTimed(obv) => obv.update(event, output),
            BuiltinFeature::TradeCountTimed(count) => count.update(event, output),
            BuiltinFeature::DayOfWeek(day_of_week) => day_of_week.update_event(event, output),
            BuiltinFeature::TimeSinceSessionOpen(session) => session.update_event(event, output),
        }
    }
}

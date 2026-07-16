use crate::Float;
use crate::features::compiler::OutputSpan;
use crate::features::event::Event;
use crate::features::indicator_vector::Feature;
use crate::vectors::FeatureVector;

pub(crate) mod day_of_week;
pub(crate) mod ema;
pub(crate) mod obv;
pub(crate) mod sma;
pub(crate) mod time_since_first_event_of_day;
pub(crate) mod trade_count;

pub use day_of_week::DayOfWeek;
pub use ema::EmaFeature;
pub use obv::ObvTimedFeature;
pub use sma::{SmaFeature, SmaTimedFeature};
pub use time_since_first_event_of_day::TimeSinceFirstEventOfDay;
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
    TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDay),
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
            BuiltinFeature::TimeSinceFirstEventOfDay(clock) => clock.update_event(event, output),
        }
    }
}

#[inline]
pub(crate) fn write_outputs<F, O>(
    span: OutputSpan,
    output: &mut O,
    mut value_at: impl FnMut(usize) -> Option<F>,
) where
    F: Float,
    O: FeatureVector<F = F>,
{
    for output_index in 0..span.count {
        if let Some(value) = value_at(output_index) {
            output.set_value_at(span.start + output_index, value);
        }
    }
}

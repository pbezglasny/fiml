use crate::Float;
use crate::features::compiler::OutputSpan;
use crate::features::event::Event;
use crate::vectors::FeatureVector;

pub(crate) mod cvd;
pub(crate) mod day_of_week;
pub(crate) mod ema;
pub(crate) mod obv;
pub(crate) mod sma;
pub(crate) mod time_since_first_event_of_day;
pub(crate) mod trade_count;

use cvd::CvdFeature;
use day_of_week::DayOfWeek;
use ema::EmaFeature;
use obv::ObvTimedFeature;
use sma::{SmaFeature, SmaTimedFeature};
use time_since_first_event_of_day::TimeSinceFirstEventOfDay;
use trade_count::TradeCountTimedFeature;

/// Closed runtime adapter enum for indicators shipped by the library.
///
/// Dispatch is a match of direct calls, with no `Box` or vtable.
pub(crate) enum IndicatorAdapter<F: Float> {
    Cvd(CvdFeature<F>),
    Sma(SmaFeature<F>),
    Ema(EmaFeature<F>),
    SmaTimed(SmaTimedFeature<F>),
    ObvTimed(ObvTimedFeature<F>),
    TradeCountTimed(TradeCountTimedFeature<F>),
    DayOfWeek(DayOfWeek),
    TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDay),
}

impl<F: Float> IndicatorAdapter<F> {
    pub(crate) fn update<O: FeatureVector<F = F>>(&mut self, event: &Event<F>, output: &mut O) {
        match self {
            Self::Cvd(cvd) => cvd.update(event, output),
            Self::Sma(sma) => sma.update(event, output),
            Self::Ema(ema) => ema.update(event, output),
            Self::SmaTimed(sma) => sma.update(event, output),
            Self::ObvTimed(obv) => obv.update(event, output),
            Self::TradeCountTimed(count) => count.update(event, output),
            Self::DayOfWeek(day_of_week) => day_of_week.update_event(event, output),
            Self::TimeSinceFirstEventOfDay(clock) => clock.update_event(event, output),
        }
    }

    pub(crate) fn observes_time(&self) -> bool {
        matches!(
            self,
            Self::SmaTimed(_) | Self::ObvTimed(_) | Self::TradeCountTimed(_)
        )
    }

    pub(crate) fn observe<O: FeatureVector<F = F>>(&mut self, event: &Event<F>, output: &mut O) {
        match self {
            Self::SmaTimed(sma) => sma.observe(event.timestamp(), output),
            Self::ObvTimed(obv) => obv.observe(event.timestamp(), output),
            Self::TradeCountTimed(count) => count.observe(event.timestamp(), output),
            _ => {}
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
        output.set_value_at(
            span.start + output_index,
            value_at(output_index).unwrap_or(F::NAN),
        );
    }
}

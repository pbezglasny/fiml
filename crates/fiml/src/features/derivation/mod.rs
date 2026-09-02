use crate::event::Event;
use crate::features::compiler::OutputSpan;
use crate::order_book::OrderBook;
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

/// Closed set of feature derivations executed by
/// [`FeatureExtractor`](crate::features::FeatureExtractor).
///
/// Each variant consumes events, updates its calculation state, and writes its
/// current values into an assigned output span. Dispatch is a match of direct
/// calls, with no `Box` or vtable.
pub(crate) enum FeatureDerivation {
    Cvd(CvdFeature),
    Sma(SmaFeature),
    Ema(EmaFeature),
    SmaTimed(SmaTimedFeature),
    ObvTimed(ObvTimedFeature),
    TradeCountTimed(TradeCountTimedFeature),
    DayOfWeek(DayOfWeek),
    TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDay),
}

impl FeatureDerivation {
    pub(crate) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_span: OutputSpan,
        output: &mut O,
    ) {
        match self {
            Self::Cvd(cvd) => cvd.update(event, output_span, output),
            Self::Sma(sma) => sma.update(event, output_span, output),
            Self::Ema(ema) => ema.update(event, output_span, output),
            Self::SmaTimed(sma) => sma.update(event, output_span, output),
            Self::ObvTimed(obv) => obv.update(event, output_span, output),
            Self::TradeCountTimed(count) => count.update(event, output_span, output),
            Self::DayOfWeek(day_of_week) => day_of_week.update(event, output_span, output),
            Self::TimeSinceFirstEventOfDay(clock) => clock.update(event, output_span, output),
        }
    }

    /// Updates a derivation from the visible state of one order book.
    ///
    /// Event-based derivations return `false`. Concrete order-book variants
    /// add their direct-dispatch arm here and return `true` after writing their
    /// output span.
    pub(crate) fn update_order_book<O: FeatureVector>(
        &mut self,
        order_book: &OrderBook,
        timestamp: i64,
        output_span: OutputSpan,
        output: &mut O,
    ) -> bool {
        let _ = (order_book, timestamp, output_span, output);
        match self {
            Self::Cvd(_)
            | Self::Sma(_)
            | Self::Ema(_)
            | Self::SmaTimed(_)
            | Self::ObvTimed(_)
            | Self::TradeCountTimed(_)
            | Self::DayOfWeek(_)
            | Self::TimeSinceFirstEventOfDay(_) => false,
        }
    }
}

#[inline]
pub(crate) fn write_outputs<O>(
    span: OutputSpan,
    output: &mut O,
    mut value_at: impl FnMut(usize) -> Option<f64>,
) where
    O: FeatureVector,
{
    for output_index in 0..span.count {
        output.set_value_at(
            span.start + output_index,
            value_at(output_index).unwrap_or(f64::NAN),
        );
    }
}

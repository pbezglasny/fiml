//! Adapts built-in event and order-book calculations to feature-vector outputs.

use crate::event::Event;
use crate::features::compiler::OutputRange;
use crate::order_book::OrderBook;
use crate::vectors::FeatureVector;

pub(crate) mod cvd;
pub(crate) mod day_of_week;
pub(crate) mod ema;
pub(crate) mod obv;
pub(crate) mod order_book;
pub(crate) mod sma;
pub(crate) mod time_since_first_event_of_day;
pub(crate) mod trade_count;
pub(crate) mod vpt;

use cvd::CvdFeature;
use day_of_week::DayOfWeek;
use ema::EmaFeature;
use obv::ObvTimedFeature;
use sma::{SmaFeature, SmaTimedFeature};
use time_since_first_event_of_day::TimeSinceFirstEventOfDay;
use trade_count::TradeCountTimedFeature;
use vpt::VptFeature;

/// Closed set of feature derivations executed by
/// [`FeatureExtractor`](crate::features::FeatureExtractor).
///
/// Each variant consumes events, updates its calculation state, and writes its
/// current values into an assigned output range. Dispatch is a match of direct
/// calls, with no `Box` or vtable.
pub(crate) enum FeatureDerivation {
    Cvd(CvdFeature),
    Sma(SmaFeature),
    Ema(EmaFeature),
    SmaTimed(SmaTimedFeature),
    ObvTimed(ObvTimedFeature),
    TradeCountTimed(TradeCountTimedFeature),
    Vpt(VptFeature),
    DayOfWeek(DayOfWeek),
    TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDay),
    OrderBook(order_book::OrderBookFeature),
}

impl FeatureDerivation {
    pub(crate) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_range: OutputRange,
        output: &mut O,
    ) {
        match self {
            Self::Cvd(cvd) => cvd.update(event, output_range, output),
            Self::Sma(sma) => sma.update(event, output_range, output),
            Self::Ema(ema) => ema.update(event, output_range, output),
            Self::SmaTimed(sma) => sma.update(event, output_range, output),
            Self::ObvTimed(obv) => obv.update(event, output_range, output),
            Self::TradeCountTimed(count) => count.update(event, output_range, output),
            Self::Vpt(vpt) => vpt.update(event, output_range, output),
            Self::DayOfWeek(day_of_week) => day_of_week.update(event, output_range, output),
            Self::TimeSinceFirstEventOfDay(clock) => clock.update(event, output_range, output),
            Self::OrderBook(_) => {}
        }
    }

    /// Updates a derivation from the visible state of one order book.
    ///
    /// Event-based derivations return `false`. Book derivations return `true`
    /// after writing their output range, including when values are missing.
    pub(crate) fn update_order_book<O: FeatureVector>(
        &mut self,
        order_book: &OrderBook,
        _timestamp: i64,
        output_range: OutputRange,
        output: &mut O,
    ) -> bool {
        match self {
            Self::OrderBook(feature) => {
                feature.update(order_book, output_range, output);
                true
            }
            Self::Cvd(_)
            | Self::Sma(_)
            | Self::Ema(_)
            | Self::SmaTimed(_)
            | Self::ObvTimed(_)
            | Self::TradeCountTimed(_)
            | Self::Vpt(_)
            | Self::DayOfWeek(_)
            | Self::TimeSinceFirstEventOfDay(_) => false,
        }
    }
}

#[inline]
pub(crate) fn write_outputs<O>(
    output_range: OutputRange,
    output: &mut O,
    mut value_at: impl FnMut(usize) -> Option<f64>,
) where
    O: FeatureVector,
{
    for output_index in 0..output_range.count {
        output.set_value_at(
            output_range.start + output_index,
            value_at(output_index).unwrap_or(f64::NAN),
        );
    }
}

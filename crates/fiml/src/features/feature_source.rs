//! Selects event fields or complete events as feature inputs.
//!
use crate::{Event, EventKind, Symbol};

/// Identifies the input stream whose observations establish a raw feature's source age.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Events of one kind; fields carried by the same event share its source time.
    Event(EventKind),
    /// The configured symbol's event clock, or the global maximum accepted event clock.
    /// This is not evidence that every market feed is fresh.
    AnyEvent,
    /// Updates applied to visible order-book state, including snapshot replay.
    OrderBook,
}

/// Source identity and last observation consumed by a raw feature.
///
/// This reports source facts, independently of output observations, numeric readiness,
/// and age policy. Consumers must check every dependency against evaluation time in
/// the same clock domain; missing, invalid, or future times cannot establish freshness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceObservation {
    /// Configured input symbol; global calendar inputs use [`Symbol::GLOBAL`].
    pub symbol: Symbol,
    /// Input kind consumed by the feature.
    pub kind: SourceKind,
    /// Original epoch-millisecond timestamp, or `None` before a relevant observation.
    /// Values are not clamped or compared against a wall clock.
    pub timestamp_millis: Option<i64>,
}

/// Selects the scalar field or complete event consumed by a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureSource {
    /// One numeric field, delivered only by its corresponding event kind.
    Field(EventField),
    /// Complete events of the selected kind, such as trades with price and volume.
    Event(EventKind),
    /// Every kind for the configured symbol. Global calendar features follow
    /// the maximum accepted timestamp across all symbols.
    AnyEvent,
}

impl FeatureSource {
    pub(crate) const fn canonical_name(self) -> &'static str {
        match self {
            Self::Field(EventField::Price) => "field.price",
            Self::Field(EventField::Volume) => "field.volume",
            Self::Field(EventField::TradePrice) => "field.trade_price",
            Self::Field(EventField::TradeVolume) => "field.trade_volume",
            Self::Event(EventKind::Price) => "event.price",
            Self::Event(EventKind::Volume) => "event.volume",
            Self::Event(EventKind::Trade) => "event.trade",
            Self::Event(EventKind::OrderBookDelta) => "event.order_book_delta",
            Self::Event(EventKind::OrderBookSnapshot) => "event.order_book_snapshot",
            Self::Event(EventKind::Time) => "event.time",
            Self::AnyEvent => "any_event",
        }
    }
}

macro_rules! define_event_field {
      (
          $(
              $(#[$meta:meta])* $source:ident => $event:ident.$field:ident
          ),+ $(,)?
      ) => {
          /// A numeric market-event field usable as an indicator input.
          #[derive(
              Debug,
              Clone,
              Copy,
              PartialEq,
              Eq,
              Hash,
              PartialOrd,
              Ord,
          )]
          #[repr(u8)]
          pub enum EventField {
              $(
                  $(#[$meta])*
                  $source,
              )+
          }

          impl EventField {
              /// Returns the event kind that carries this field.
              pub const fn event_kind(self) -> EventKind {
                  match self {
                      $(
                          Self::$source => EventKind::$event,
                      )+
                  }
              }

              /// Reads the field, or returns `None` when the event kind does not match.
              pub fn extract(self, event: &Event) -> Option<f64>
              {
                  match (self, event) {
                      $(
                          (
                              Self::$source,
                              Event::$event(update),
                          ) => Some(update.$field),
                      )+
                      _ => None,
                  }
              }
          }
      };
  }

define_event_field! {
    /// Price from a standalone price update.
    Price => Price.value,
    /// Volume from a standalone volume update.
    Volume => Volume.value,
    /// Execution price from a trade.
    TradePrice => Trade.price,
    /// Executed volume from a trade.
    TradeVolume => Trade.volume
}

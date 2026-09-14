//! Selects event fields or complete events as feature inputs.
//!
use crate::{Event, EventKind};

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

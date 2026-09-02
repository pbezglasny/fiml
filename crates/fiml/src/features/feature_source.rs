use crate::{Event, EventKind};

/// Source of value to calculate feature
/// Each event could provide multiple sources or
/// could be entire source of feature.
/// E.g. Trade can provide trade price, trade volume,
/// or could be use whole
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureSource {
    Field(EventField),
    Event(EventKind),
    EveryEvent,
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
            Self::EveryEvent => "every_event",
        }
    }
}

macro_rules! define_event_field {
      (
          $(
              $source:ident => $event:ident.$field:ident
          ),+ $(,)?
      ) => {
          /// Use next field of event as argument of indicator
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
                  $source,
              )+
          }

          impl EventField {
              pub const fn event_kind(self) -> EventKind {
                  match self {
                      $(
                          Self::$source => EventKind::$event,
                      )+
                  }
              }

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
    Price => Price.value,
    Volume => Volume.value,
    TradePrice => Trade.price,
    TradeVolume => Trade.volume
}

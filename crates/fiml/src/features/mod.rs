mod builder;
pub(crate) mod builtin;
pub(crate) mod compiler;
mod definition;
mod event;
mod extractor;
pub(crate) mod indicator_vector;
mod pipeline;
#[cfg(feature = "serde")]
mod serialization;
pub mod transformers;

pub use builder::FeatureSetBuilder;
pub use definition::{
    FeatureSet, IndicatorSpec, MAX_OUTPUTS_PER_INDICATOR, ScopedIndicator, TimeWindows, ValueSource,
};
pub use event::{
    EVENT_KIND_COUNT, Event, EventKind, OrderBookUpdate, PriceUpdate, TimeUpdate, TradeSide,
    TradeUpdate, VolumeUpdate,
};
pub use extractor::{DispatchSequenceError, FeatureExtractor};
pub use indicator_vector::{IndicatorFeatureVector, IndicatorFeatures};
pub use pipeline::Pipeline;
#[cfg(feature = "serde")]
pub use serialization::FEATURE_SET_FORMAT_VERSION;

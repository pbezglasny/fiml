mod builder;
pub(crate) mod builtin;
pub(crate) mod compiler;
mod definition;
mod event;
mod extractor;
pub(crate) mod indicator_vector;
mod pipeline;
pub mod transformers;

pub use builder::FeatureSetBuilder;
pub use builtin::BuiltinFeature;
pub use builtin::{DayOfWeek, EmaFeature, ObvTimedFeature, SmaFeature, SmaTimedFeature};
#[cfg(feature = "serde")]
pub use definition::FEATURE_SET_FORMAT_VERSION;
pub use definition::{
    FeatureSet, IndicatorDef, IndicatorSpec, MAX_OUTPUTS_PER_INDICATOR, TimeWindows, ValueSource,
};
pub use event::{
    EVENT_KIND_COUNT, Event, EventKind, OrderBookUpdate, PriceUpdate, TimeUpdate, TradeUpdate,
    VolumeUpdate,
};
pub use extractor::{DispatchSequenceError, FeatureExtractor};
pub use indicator_vector::{Feature, IndicatorFeatureVector, IndicatorFeatures};
pub use pipeline::Pipeline;

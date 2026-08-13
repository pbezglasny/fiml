mod builder;
pub(crate) mod builtin;
pub(crate) mod compiler;
mod definition;
mod extractor;
pub(crate) mod indicator_vector;
mod pipeline;
pub mod transformers;

use crate::event::{EVENT_KIND_COUNT, EventKind};

/// Number of dispatch groups: one per [`EventKind`] plus the group that runs
/// on every event.
pub(crate) const FEATURE_GROUP_COUNT: usize = EVENT_KIND_COUNT + 1;

/// Index of the every-event group in the feature dispatch table.
pub(crate) const EVERY_EVENT_GROUP: usize = EVENT_KIND_COUNT;

/// Where a feature subscribes in the dispatch table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatureRoute {
    Kind(EventKind),
    Every,
}

impl FeatureRoute {
    pub(crate) fn group_index(self) -> usize {
        match self {
            Self::Kind(kind) => kind as usize,
            Self::Every => EVERY_EVENT_GROUP,
        }
    }
}

pub use builder::FeatureSetBuilder;
pub use definition::{
    FeatureSet, IndicatorSpec, MAX_OUTPUTS_PER_INDICATOR, ScopedIndicator, TimeWindows, ValueSource,
};
pub use extractor::{DispatchSequenceError, FeatureExtractor};
pub use indicator_vector::{IndicatorFeatureVector, IndicatorFeatures};
pub use pipeline::Pipeline;

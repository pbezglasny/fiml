mod builder;
pub(crate) mod builtin;
pub(crate) mod compiler;
mod definition;
mod feature_id;
mod feature_key;
mod feature_source;
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
pub use feature_id::FeatureId;
pub use feature_key::FeatureKey;
pub use feature_source::{EventField, FeatureSource};

/// Declaration of one scalar output in a feature vector.
///
/// [`FeatureKey`] describes the calculation that produces the value, while
/// [`FeatureId`] provides its stable user-facing name. Compatible definitions
/// may be grouped into one runtime indicator; grouping does not change the ID
/// used to locate each definition's scalar output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureDefinition {
    /// Complete structural identity of the calculation and scalar output.
    pub key: FeatureKey,
    /// Name used to find this output in a compiled feature-vector layout.
    pub id: FeatureId,
}

impl FeatureDefinition {
    /// Creates a feature definition with an explicit user-facing ID.
    pub fn new(key: FeatureKey, id: FeatureId) -> Self {
        Self { key, id }
    }

    /// Creates a feature definition whose ID is derived from its key.
    pub fn with_default_id(key: FeatureKey) -> Self {
        let id = FeatureId::from_feature_key(&key);
        Self { key, id }
    }
}

//! Defines feature configuration, compilation, routing, and extraction.
//!
//! The module's intent is to provide one construction path from scalar feature
//! definitions to an allocation-free event-processing runtime. Public types
//! describe feature identity and output storage, while the compiler,
//! derivations, and router remain implementation details behind
//! [`FeatureExtractor`].

pub(crate) mod compiler;
pub(crate) mod derivation;
mod feature_extractor;
mod feature_extractor_builder;
mod feature_id;
mod feature_key;
mod feature_source;
mod pipeline;
mod serde;

use crate::event::EventKind;

/// Maximum number of adjacent outputs one runtime derivation may own.
pub const MAX_OUTPUTS_PER_INDICATOR: usize = 16;

/// Where a feature subscribes in the dispatch table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatureRoute {
    Kind(EventKind),
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed when the first built-in order-book derivation is compiled"
        )
    )]
    OrderBook,
    Any,
}

pub use feature_extractor::{FeatureExtractor, UpdateResult};
pub use feature_extractor_builder::FeatureExtractorBuilder;
pub use feature_id::FeatureId;
pub use feature_key::FeatureKey;
pub use feature_source::{EventField, FeatureSource};
pub use pipeline::{ModelInputSpec, Pipeline, TransformationDefinition};
pub use serde::FeatureVectorSpec;

const RESERVED_ID_PREFIX: &str = "__reserved_";

pub(crate) fn is_reserved_feature_id(id: &FeatureId) -> bool {
    id.as_str().starts_with(RESERVED_ID_PREFIX)
}

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

mod feature;
mod feature_set;
mod indicator;
mod scalar;

use serde::{Deserialize, Serialize};

use super::definition::FeatureSet;
use feature_set::{FeatureSetRef, FeatureSetWire};

/// Semantic version emitted in serialized [`FeatureSet`] JSON artifacts.
pub const FEATURE_SET_FORMAT_VERSION: &str = "1.0.0";

impl Serialize for FeatureSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        FeatureSetRef::new(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FeatureSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        FeatureSetWire::deserialize(deserializer)?
            .into_feature_set()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::FEATURE_SET_FORMAT_VERSION;
use super::feature::{FeatureGroup, FeatureGroupsRef};
use super::indicator::options::EmptyOptions;
use crate::features::definition::FeatureSet;

#[derive(Serialize)]
pub(super) struct FeatureSetRef<'a> {
    version: &'static str,
    features: FeatureGroupsRef<'a>,
    options: EmptyOptions,
}

impl<'a> FeatureSetRef<'a> {
    pub(super) fn new(feature_set: &'a FeatureSet) -> Self {
        Self {
            version: FEATURE_SET_FORMAT_VERSION,
            features: FeatureGroupsRef::new(&feature_set.indicators),
            options: EmptyOptions {},
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FeatureSetWire {
    version: String,
    features: Vec<FeatureGroup>,
    #[serde(rename = "options")]
    _options: EmptyOptions,
}

impl FeatureSetWire {
    pub(super) fn into_feature_set(self) -> Result<FeatureSet, String> {
        validate_feature_set_version(&self.version)?;

        let mut definitions = Vec::new();
        let mut scopes = HashSet::with_capacity(self.features.len());
        for (index, feature) in self.features.into_iter().enumerate() {
            let (scope, mut group_definitions) = feature.into_definitions(index)?;
            if !scopes.insert(scope.clone()) {
                let scope = scope
                    .as_deref()
                    .map(|symbol| format!("symbol {symbol:?}"))
                    .unwrap_or_else(|| "global scope".to_owned());
                return Err(format!(
                    "duplicate feature group for {scope} after symbol normalization"
                ));
            }
            definitions.append(&mut group_definitions);
        }

        Ok(FeatureSet::new(definitions))
    }
}

fn parse_feature_set_version(version: &str) -> Result<semver::Version, semver::Error> {
    match semver::Version::parse(version) {
        Ok(version) => Ok(version),
        Err(original_error) => {
            let suffix_start = version.find(['-', '+']).unwrap_or(version.len());
            let (core, suffix) = version.split_at(suffix_start);
            if core.bytes().filter(|byte| *byte == b'.').count() != 1 {
                return Err(original_error);
            }

            semver::Version::parse(&format!("{core}.0{suffix}"))
        }
    }
}

fn feature_set_versions_are_compatible(
    artifact: &semver::Version,
    supported: &semver::Version,
) -> bool {
    artifact.major == supported.major
        && artifact.minor == supported.minor
        && artifact.pre.is_empty()
        && supported.pre.is_empty()
}

fn validate_feature_set_version(version: &str) -> Result<(), String> {
    let artifact = parse_feature_set_version(version)
        .map_err(|error| format!("invalid feature set version {version:?}: {error}"))?;
    let supported = semver::Version::parse(FEATURE_SET_FORMAT_VERSION)
        .expect("the feature set format version constant must be valid SemVer");
    if feature_set_versions_are_compatible(&artifact, &supported) {
        Ok(())
    } else {
        Err(format!(
            "unsupported feature set version {version:?}; this library supports stable version {}.{}.x",
            supported.major, supported.minor
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compatibility_accepts_current_minor_patch_versions() {
        for version in ["1.0", "1.0.0", "1.0.9", "1.0.0+training-run"] {
            assert!(validate_feature_set_version(version).is_ok(), "{version}");
        }
    }

    #[test]
    fn version_compatibility_rejects_future_minor_and_prerelease_versions() {
        for version in ["1.1.0", "2.0.0", "1.0.0-beta.1"] {
            assert!(
                validate_feature_set_version(version)
                    .unwrap_err()
                    .contains("unsupported feature set version"),
                "{version}"
            );
        }
    }
}

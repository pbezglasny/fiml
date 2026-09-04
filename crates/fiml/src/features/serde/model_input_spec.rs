use ::serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{FeatureVectorSpec, serialization::deserialize_present_option};
use crate::{FeatureId, ModelInputSpec, TransformationDefinition};

const FORMAT_VERSION: &str = "1.0";

/// Private versioned storage representation for a complete model-input layout.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInputSpecWire {
    version: String,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    checksum: Option<String>,
    feature_extractor: FeatureVectorSpec,
    model_input: ModelInputWire,
}

/// Private storage representation for the final model-vector layout.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInputWire {
    capacity: usize,
    length: usize,
    transformations: Vec<TransformationWire>,
}

/// Private ID-based representation of the supported scalar transformations.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum TransformationWire {
    Identity {
        input: String,
        output: String,
    },
    StandardScale {
        input: String,
        output: String,
        mean: f64,
        scale: f64,
    },
}

impl Serialize for ModelInputSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModelInputSpecWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ModelInputSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModelInputSpecWire::deserialize(deserializer)?;
        ModelInputSpec::try_from(wire).map_err(::serde::de::Error::custom)
    }
}

impl From<&ModelInputSpec> for ModelInputSpecWire {
    fn from(spec: &ModelInputSpec) -> Self {
        Self {
            version: FORMAT_VERSION.to_owned(),
            checksum: spec.checksum().map(str::to_owned),
            feature_extractor: spec.raw_feature_vector_spec().clone(),
            model_input: ModelInputWire {
                capacity: spec.feature_vector_capacity(),
                length: spec.feature_vector_length(),
                transformations: spec
                    .transformation_definitions()
                    .iter()
                    .map(TransformationWire::from)
                    .collect(),
            },
        }
    }
}

impl From<&TransformationDefinition> for TransformationWire {
    fn from(definition: &TransformationDefinition) -> Self {
        match definition {
            TransformationDefinition::Identity { input, output } => Self::Identity {
                input: input.as_str().to_owned(),
                output: output.as_str().to_owned(),
            },
            TransformationDefinition::StandardScale {
                input,
                output,
                mean,
                scale,
            } => Self::StandardScale {
                input: input.as_str().to_owned(),
                output: output.as_str().to_owned(),
                mean: *mean,
                scale: *scale,
            },
        }
    }
}

impl TryFrom<ModelInputSpecWire> for ModelInputSpec {
    type Error = String;

    fn try_from(wire: ModelInputSpecWire) -> Result<Self, Self::Error> {
        if wire.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported model-input spec version {:?}; expected {FORMAT_VERSION:?}",
                wire.version
            ));
        }
        if wire.model_input.length != wire.model_input.transformations.len() {
            return Err(format!(
                "model_input.length {} does not match transformation count {}",
                wire.model_input.length,
                wire.model_input.transformations.len()
            ));
        }
        if wire.model_input.capacity < wire.model_input.length {
            return Err(format!(
                "model_input.capacity {} is smaller than model_input.length {}",
                wire.model_input.capacity, wire.model_input.length
            ));
        }

        let transformations = wire
            .model_input
            .transformations
            .into_iter()
            .map(TransformationDefinition::from)
            .collect::<Vec<_>>();
        ModelInputSpec::with_metadata(
            wire.feature_extractor,
            transformations,
            wire.model_input.capacity,
            wire.checksum,
        )
        .map_err(|error| error.to_string())
    }
}

impl From<TransformationWire> for TransformationDefinition {
    fn from(transformation: TransformationWire) -> Self {
        match transformation {
            TransformationWire::Identity { input, output } => {
                Self::identity(FeatureId::new(input), FeatureId::new(output))
            }
            TransformationWire::StandardScale {
                input,
                output,
                mean,
                scale,
            } => Self::standard_scale(FeatureId::new(input), FeatureId::new(output), mean, scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::{FeatureDefinition, FeatureKey, FeatureSource, Symbol};

    use super::*;

    fn day_of_week(id: &str) -> FeatureDefinition {
        FeatureDefinition::new(
            FeatureKey::DayOfWeek {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::AnyEvent,
            },
            FeatureId::new(id),
        )
    }

    fn time_since_first_event(id: &str) -> FeatureDefinition {
        FeatureDefinition::new(
            FeatureKey::TimeSinceFirstEventOfDay {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::AnyEvent,
                utc_offset_millis: 0,
            },
            FeatureId::new(id),
        )
    }

    fn valid_model_spec() -> Value {
        json!({
            "version": "1.0",
            "feature_extractor": {
                "version": "1.0",
                "capacity": 1,
                "length": 1,
                "features": [{
                    "symbol": "__global__",
                    "indicators": [{
                        "kind": "day_of_week",
                        "source": {"type": "any_event"},
                        "outputs": [{"id": "raw_day"}]
                    }]
                }]
            },
            "model_input": {
                "capacity": 1,
                "length": 1,
                "transformations": [{
                    "type": "identity",
                    "input": "raw_day",
                    "output": "day"
                }]
            }
        })
    }

    fn error(value: Value) -> String {
        serde_json::from_value::<ModelInputSpec>(value)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn round_trip_writes_canonical_contract_and_preserves_transformation_order() {
        let raw_spec = FeatureVectorSpec::with_metadata(
            [
                time_since_first_event("raw_elapsed"),
                day_of_week("raw_day"),
            ],
            3,
            Some("raw-checksum".to_owned()),
        )
        .unwrap();
        let spec = ModelInputSpec::with_metadata(
            raw_spec,
            [
                TransformationDefinition::standard_scale(
                    FeatureId::new("raw_elapsed"),
                    FeatureId::new("scaled_elapsed"),
                    4.0,
                    2.0,
                ),
                TransformationDefinition::identity(
                    FeatureId::new("raw_day"),
                    FeatureId::new("day"),
                ),
            ],
            4,
            Some("model-checksum".to_owned()),
        )
        .unwrap();

        let value = serde_json::to_value(&spec).unwrap();
        assert_eq!(
            value,
            json!({
                "version": "1.0",
                "checksum": "model-checksum",
                "feature_extractor": {
                    "version": "1.0",
                    "capacity": 3,
                    "length": 2,
                    "checksum": "raw-checksum",
                    "features": [{
                        "symbol": "__global__",
                        "indicators": [
                            {
                                "kind": "day_of_week",
                                "source": {"type": "any_event"},
                                "outputs": [{"id": "raw_day"}]
                            },
                            {
                                "kind": "time_since_first_event_of_day",
                                "source": {"type": "any_event"},
                                "options": {"utc_offset": "+00:00"},
                                "outputs": [{"id": "raw_elapsed"}]
                            }
                        ]
                    }]
                },
                "model_input": {
                    "capacity": 4,
                    "length": 2,
                    "transformations": [
                        {
                            "type": "standard_scale",
                            "input": "raw_elapsed",
                            "output": "scaled_elapsed",
                            "mean": 4.0,
                            "scale": 2.0
                        },
                        {
                            "type": "identity",
                            "input": "raw_day",
                            "output": "day"
                        }
                    ]
                }
            })
        );

        let restored: ModelInputSpec = serde_json::from_value(value).unwrap();
        assert_eq!(restored, spec);
    }

    #[test]
    fn documented_example_is_accepted_and_already_canonical() {
        let text = include_str!("../../../../../docs/example_of_store_definition.json");
        let documented: Value = serde_json::from_str(text).unwrap();
        let spec: ModelInputSpec = serde_json::from_str(text).unwrap();

        assert_eq!(serde_json::to_value(spec).unwrap(), documented);
    }

    #[test]
    fn absent_checksums_are_omitted_and_explicit_null_is_rejected() {
        let spec: ModelInputSpec = serde_json::from_value(valid_model_spec()).unwrap();
        let value = serde_json::to_value(spec).unwrap();

        assert!(value.get("checksum").is_none());
        assert!(value["feature_extractor"].get("checksum").is_none());

        let mut value = valid_model_spec();
        value["checksum"] = Value::Null;
        assert!(error(value).contains("string"));

        let mut value = valid_model_spec();
        value["feature_extractor"]["checksum"] = Value::Null;
        assert!(error(value).contains("string"));
    }

    #[test]
    fn rejects_unsupported_versions_and_dimension_mismatches() {
        let mut value = valid_model_spec();
        value["version"] = json!("2.0");
        assert!(error(value).contains("unsupported model-input spec version"));

        let mut value = valid_model_spec();
        value["feature_extractor"]["version"] = json!("2.0");
        assert!(error(value).contains("unsupported feature-vector spec version"));

        let mut value = valid_model_spec();
        value["model_input"]["length"] = json!(2);
        assert!(error(value).contains("does not match transformation count"));

        let mut value = valid_model_spec();
        value["model_input"]["capacity"] = json!(0);
        assert!(error(value).contains("smaller than model_input.length"));
    }

    #[test]
    fn rejects_unknown_and_missing_envelope_fields() {
        let mut value = valid_model_spec();
        value["unknown"] = json!(true);
        assert!(error(value).contains("unknown field"));

        let mut value = valid_model_spec();
        let feature_extractor = value
            .as_object_mut()
            .unwrap()
            .remove("feature_extractor")
            .unwrap();
        value["raw_feature_vector_spec"] = feature_extractor;
        assert!(error(value).contains("unknown field `raw_feature_vector_spec`"));

        for field in ["version", "feature_extractor", "model_input"] {
            let mut value = valid_model_spec();
            value.as_object_mut().unwrap().remove(field);
            assert!(
                error(value).contains("missing field"),
                "missing {field} should be rejected"
            );
        }

        let mut value = valid_model_spec();
        value["model_input"]["unknown"] = json!(true);
        assert!(error(value).contains("unknown field"));

        for field in ["capacity", "length", "transformations"] {
            let mut value = valid_model_spec();
            value["model_input"].as_object_mut().unwrap().remove(field);
            assert!(
                error(value).contains("missing field"),
                "missing model_input.{field} should be rejected"
            );
        }

        let mut value = valid_model_spec();
        let model_input = value
            .as_object_mut()
            .unwrap()
            .remove("model_input")
            .unwrap();
        value["feature_vector_capacity"] = model_input["capacity"].clone();
        value["feature_vector_length"] = model_input["length"].clone();
        value["transformations"] = model_input["transformations"].clone();
        assert!(error(value).contains("unknown field"));
    }

    #[test]
    fn rejects_unknown_and_malformed_transformation_variants() {
        let mut value = valid_model_spec();
        value["model_input"]["transformations"][0]["type"] = json!("normalize");
        assert!(error(value).contains("unknown variant"));

        let mut value = valid_model_spec();
        value["model_input"]["transformations"][0]["extra"] = json!(1);
        assert!(error(value).contains("unknown field"));

        let mut value = valid_model_spec();
        value["model_input"]["transformations"][0] = json!({
            "type": "standard_scale",
            "input": "raw_day",
            "output": "day",
            "mean": 4.0
        });
        assert!(error(value).contains("missing field `scale`"));

        let mut value = valid_model_spec();
        value["model_input"]["transformations"] = Value::Null;
        assert!(error(value).contains("sequence"));
    }

    #[test]
    fn deserialization_reuses_model_input_semantic_validation() {
        let mut value = valid_model_spec();
        value["model_input"]["transformations"][0]["input"] = json!("missing");
        assert!(error(value).contains("input feature ID does not exist"));

        let mut value = valid_model_spec();
        value["model_input"]["capacity"] = json!(2);
        value["model_input"]["length"] = json!(2);
        value["model_input"]["transformations"] = json!([
            {"type":"identity", "input":"raw_day", "output":"day"},
            {"type":"identity", "input":"raw_day", "output":"day"}
        ]);
        assert!(error(value).contains("output feature ID duplicates an earlier output"));

        let mut value = valid_model_spec();
        value["model_input"]["transformations"][0]["output"] = json!("__reserved_0");
        assert!(error(value).contains("reserved namespace"));

        for scale in [json!(0.0), json!(-1.0)] {
            let mut value = valid_model_spec();
            value["model_input"]["transformations"][0] = json!({
                "type":"standard_scale", "input":"raw_day", "output":"day",
                "mean":0.0, "scale":scale
            });
            assert!(error(value).contains("scale must be positive"));
        }

        let mut value = valid_model_spec();
        value["model_input"]["transformations"][0] = json!({
            "type":"standard_scale", "input":"raw_day", "output":"day",
            "mean":0.0, "scale":5e-324
        });
        assert!(error(value).contains("inverse scale must be finite"));

        for text in [
            r#"{"version":"1.0","feature_extractor":{"version":"1.0","capacity":0,"length":0,"features":[]},"model_input":{"capacity":1,"length":1,"transformations":[{"type":"standard_scale","input":"raw_day","output":"day","mean":NaN,"scale":1.0}]}}"#,
            r#"{"version":"1.0","feature_extractor":{"version":"1.0","capacity":0,"length":0,"features":[]},"model_input":{"capacity":1,"length":1,"transformations":[{"type":"standard_scale","input":"raw_day","output":"day","mean":0.0,"scale":1e400}]}}"#,
        ] {
            assert!(serde_json::from_str::<ModelInputSpec>(text).is_err());
        }
    }
}

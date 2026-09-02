//! Compiles raw feature extraction and scalar transformations into model input.
//!
//! The spec types validate named layouts on the cold path. [`Pipeline`] keeps
//! only resolved indexes and writes one caller-owned model vector directly on
//! the event-processing hot path.

mod specs;

pub use specs::{ModelInputSpec, TransformationDefinition};

use crate::{Event, FeatureExtractor, FeatureId, FeatureVector, Float, Result, UpdateResult};

enum ScalarOperation<F> {
    Identity {
        input_index: usize,
        output_index: usize,
    },
    StandardScale {
        input_index: usize,
        output_index: usize,
        mean: F,
        inverse_scale: F,
    },
}

/// Allocation-free event runtime for raw extraction and final model input.
pub struct Pipeline<F, RawV, ModelV>
where
    F: Float,
    RawV: FeatureVector<F = F>,
    ModelV: FeatureVector<F = F>,
{
    feature_extractor: FeatureExtractor<F, RawV>,
    operations: Box<[ScalarOperation<F>]>,
    model_vector: ModelV,
    output_ids: Box<[FeatureId]>,
}

impl<F, RawV, ModelV> Pipeline<F, RawV, ModelV>
where
    F: Float,
    RawV: FeatureVector<F = F>,
    ModelV: FeatureVector<F = F>,
{
    /// Applies an accepted event to raw features and then refreshes model input.
    #[must_use = "event errors must be handled before using updated model-input values"]
    pub fn handle_event(&mut self, event: Event<F>) -> Result<UpdateResult> {
        let update_result = self.feature_extractor.handle_event(event)?;
        let raw_values = self.feature_extractor.feature_vector().values();
        for operation in &self.operations {
            match *operation {
                ScalarOperation::Identity {
                    input_index,
                    output_index,
                } => self
                    .model_vector
                    .set_value_at(output_index, raw_values[input_index]),
                ScalarOperation::StandardScale {
                    input_index,
                    output_index,
                    mean,
                    inverse_scale,
                } => self.model_vector.set_value_at(
                    output_index,
                    (raw_values[input_index] - mean) * inverse_scale,
                ),
            }
        }
        Ok(update_result)
    }

    /// Returns the raw extractor vector, including reserved cells.
    pub fn raw_values(&self) -> &[F] {
        self.feature_extractor.feature_vector().values()
    }

    /// Returns final model input, including reserved cells.
    pub fn values(&self) -> &[F] {
        self.model_vector.values()
    }

    /// Returns active final IDs in model-vector order.
    pub fn output_ids(&self) -> &[FeatureId] {
        &self.output_ids
    }

    /// Returns the timestamp of the last accepted event.
    pub fn last_timestamp(&self) -> Option<i64> {
        self.feature_extractor.last_timestamp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArrayFeatureVector, EventField, FeatureDefinition, FeatureKey, FeatureSource,
        FeatureVectorSpec, FimlError, InvalidArgumentError, InvalidTransformationDefinitionError,
        Symbol, WarmupPolicy,
    };

    fn day_of_week(id: &str) -> FeatureDefinition {
        FeatureDefinition::new(
            FeatureKey::DayOfWeek {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::EveryEvent,
            },
            FeatureId::new(id),
        )
    }

    fn time_since_first_event(id: &str) -> FeatureDefinition {
        FeatureDefinition::new(
            FeatureKey::TimeSinceFirstEventOfDay {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::EveryEvent,
                utc_offset_millis: 0,
            },
            FeatureId::new(id),
        )
    }

    fn warming_sma(id: &str) -> FeatureDefinition {
        FeatureDefinition::new(
            FeatureKey::Sma {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::Field(EventField::Price),
                window: 2,
                warmup_policy: WarmupPolicy::FullWindow,
            },
            FeatureId::new(id),
        )
    }

    macro_rules! test_scalar_transformations {
        ($name:ident, $float:ty) => {
            #[test]
            fn $name() {
                let raw_spec = FeatureVectorSpec::new([day_of_week("day")]).unwrap();
                let spec = ModelInputSpec::new(
                    raw_spec,
                    [
                        TransformationDefinition::identity(
                            FeatureId::new("day"),
                            FeatureId::new("identity"),
                        ),
                        TransformationDefinition::standard_scale(
                            FeatureId::new("day"),
                            FeatureId::new("scaled"),
                            2.0 as $float,
                            2.0 as $float,
                        ),
                    ],
                )
                .unwrap();
                let mut pipeline = spec
                    .build(
                        ArrayFeatureVector::<$float, 1>::new(),
                        ArrayFeatureVector::<$float, 2>::new(),
                    )
                    .unwrap();

                assert!(pipeline.values().iter().all(|value| value.is_nan()));
                pipeline.handle_event(Event::time(0)).unwrap();

                assert_eq!(pipeline.raw_values(), &[4.0 as $float]);
                assert_eq!(pipeline.values(), &[4.0 as $float, 1.0 as $float]);
            }
        };
    }

    test_scalar_transformations!(identity_and_standard_scaling_work_for_f32, f32);
    test_scalar_transformations!(identity_and_standard_scaling_work_for_f64, f64);

    #[test]
    fn authored_order_is_final_order_and_raw_and_final_ids_are_separate() {
        let raw_spec =
            FeatureVectorSpec::new([time_since_first_event("elapsed"), day_of_week("day")])
                .unwrap();
        let spec = ModelInputSpec::with_metadata(
            raw_spec,
            [
                TransformationDefinition::identity(
                    FeatureId::new("elapsed"),
                    FeatureId::new("elapsed"),
                ),
                TransformationDefinition::identity(
                    FeatureId::new("day"),
                    FeatureId::new("model_day"),
                ),
            ],
            4,
            Some("opaque".to_owned()),
        )
        .unwrap();

        assert_eq!(spec.feature_vector_length(), 2);
        assert_eq!(spec.feature_vector_capacity(), 4);
        assert_eq!(spec.checksum(), Some("opaque"));
        assert_eq!(spec.raw_feature_vector_spec().feature_vector_length(), 2);
        assert_eq!(spec.transformation_definitions().len(), 2);

        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<f64, 2>::new(),
                ArrayFeatureVector::<f64, 4>::new_of_length(2),
            )
            .unwrap();
        pipeline.handle_event(Event::time(0)).unwrap();

        assert_eq!(pipeline.output_ids()[0], FeatureId::new("elapsed"));
        assert_eq!(pipeline.output_ids()[1], FeatureId::new("model_day"));
        assert_eq!(pipeline.values()[..2], [0.0, 4.0]);
        assert!(pipeline.values()[2..].iter().all(|value| value.is_nan()));
    }

    #[test]
    fn undeclared_raw_features_are_omitted_and_scaling_does_not_mutate_raw_values() {
        let raw_spec =
            FeatureVectorSpec::new([day_of_week("day"), time_since_first_event("elapsed")])
                .unwrap();
        let spec = ModelInputSpec::new(
            raw_spec,
            [TransformationDefinition::standard_scale(
                FeatureId::new("day"),
                FeatureId::new("day"),
                2.0,
                2.0,
            )],
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<f64, 2>::new(),
                ArrayFeatureVector::<f64, 1>::new(),
            )
            .unwrap();

        pipeline.handle_event(Event::time(0)).unwrap();

        assert_eq!(pipeline.raw_values(), &[4.0, 0.0]);
        assert_eq!(pipeline.values(), &[1.0]);
        assert_eq!(pipeline.output_ids(), &[FeatureId::new("day")]);
    }

    #[test]
    fn warmup_nan_propagates_and_rejected_event_leaves_final_output_unchanged() {
        let raw_spec = FeatureVectorSpec::new([warming_sma("sma")]).unwrap();
        let spec = ModelInputSpec::new(
            raw_spec,
            [TransformationDefinition::standard_scale(
                FeatureId::new("sma"),
                FeatureId::new("scaled_sma"),
                10.0,
                5.0,
            )],
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<f64, 1>::new(),
                ArrayFeatureVector::<f64, 1>::new(),
            )
            .unwrap();

        pipeline
            .handle_event(Event::price(Symbol::GLOBAL, 10.0, 1))
            .unwrap();
        assert!(pipeline.raw_values()[0].is_nan());
        assert!(pipeline.values()[0].is_nan());

        pipeline
            .handle_event(Event::price(Symbol::GLOBAL, 20.0, 2))
            .unwrap();
        assert_eq!(pipeline.raw_values(), &[15.0]);
        assert_eq!(pipeline.values(), &[1.0]);
        let error = match pipeline.handle_event(Event::price(Symbol::GLOBAL, 100.0, 0)) {
            Err(error) => error,
            Ok(_) => panic!("out-of-order event should fail"),
        };

        assert!(matches!(error, FimlError::TimestampOutOfOrder { .. }));
        assert_eq!(pipeline.raw_values(), &[15.0]);
        assert_eq!(pipeline.values(), &[1.0]);
        assert_eq!(pipeline.last_timestamp(), Some(2));
    }

    #[test]
    fn rejects_unknown_duplicate_and_reserved_ids_and_insufficient_capacity() {
        let unknown = ModelInputSpec::<f64>::new(
            FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
            [TransformationDefinition::identity(
                FeatureId::new("missing"),
                FeatureId::new("output"),
            )],
        )
        .unwrap_err();
        assert!(matches!(
            unknown,
            FimlError::InvalidTransformationDefinition {
                index: 0,
                reason: InvalidTransformationDefinitionError::InputFeatureNotFound
            }
        ));

        let duplicate = ModelInputSpec::<f64>::new(
            FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
            [
                TransformationDefinition::identity(FeatureId::new("day"), FeatureId::new("output")),
                TransformationDefinition::identity(FeatureId::new("day"), FeatureId::new("output")),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            duplicate,
            FimlError::InvalidTransformationDefinition {
                index: 1,
                reason: InvalidTransformationDefinitionError::DuplicateOutputFeature
            }
        ));

        let reserved = ModelInputSpec::<f64>::new(
            FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
            [TransformationDefinition::identity(
                FeatureId::new("day"),
                FeatureId::new("__reserved_0"),
            )],
        )
        .unwrap_err();
        assert!(matches!(
            reserved,
            FimlError::InvalidTransformationDefinition {
                index: 0,
                reason: InvalidTransformationDefinitionError::ReservedOutputFeature
            }
        ));

        let capacity = ModelInputSpec::<f64>::with_capacity(
            FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
            [TransformationDefinition::identity(
                FeatureId::new("day"),
                FeatureId::new("output"),
            )],
            0,
        )
        .unwrap_err();
        assert!(matches!(
            capacity,
            FimlError::InvalidArgument(InvalidArgumentError::FeatureVectorCapacityTooSmall {
                capacity: 0,
                active_length: 1
            })
        ));
    }

    #[test]
    fn rejects_invalid_scaler_parameters() {
        let cases = [
            (
                f64::NAN,
                1.0,
                InvalidTransformationDefinitionError::MeanNotFinite,
            ),
            (
                0.0,
                f64::INFINITY,
                InvalidTransformationDefinitionError::ScaleNotFinite,
            ),
            (
                0.0,
                0.0,
                InvalidTransformationDefinitionError::ScaleNotPositive,
            ),
            (
                0.0,
                -1.0,
                InvalidTransformationDefinitionError::ScaleNotPositive,
            ),
            (
                0.0,
                f64::from_bits(1),
                InvalidTransformationDefinitionError::InverseScaleNotFinite,
            ),
        ];

        for (mean, scale, expected) in cases {
            let error = ModelInputSpec::new(
                FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
                [TransformationDefinition::standard_scale(
                    FeatureId::new("day"),
                    FeatureId::new("output"),
                    mean,
                    scale,
                )],
            )
            .unwrap_err();
            assert!(matches!(
                error,
                FimlError::InvalidTransformationDefinition { index: 0, reason }
                    if reason == expected
            ));
        }
    }

    #[test]
    fn rejects_raw_and_model_storage_mismatches() {
        let make_spec = || {
            ModelInputSpec::<f64>::with_capacity(
                FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
                [TransformationDefinition::identity(
                    FeatureId::new("day"),
                    FeatureId::new("output"),
                )],
                2,
            )
            .unwrap()
        };

        let capacity_error = match make_spec().build(
            ArrayFeatureVector::<f64, 1>::new(),
            ArrayFeatureVector::<f64, 1>::new(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("model capacity mismatch should fail"),
        };
        assert!(matches!(
            capacity_error,
            FimlError::ModelVectorCapacityMismatch {
                expected: 2,
                actual: 1
            }
        ));

        let length_error = match make_spec().build(
            ArrayFeatureVector::<f64, 1>::new(),
            ArrayFeatureVector::<f64, 2>::new_of_length(0),
        ) {
            Err(error) => error,
            Ok(_) => panic!("model length mismatch should fail"),
        };
        assert!(matches!(
            length_error,
            FimlError::ModelVectorLengthMismatch {
                expected: 1,
                actual: 0
            }
        ));

        let raw_error = match make_spec().build(
            ArrayFeatureVector::<f64, 2>::new_of_length(1),
            ArrayFeatureVector::<f64, 2>::new_of_length(1),
        ) {
            Err(error) => error,
            Ok(_) => panic!("raw capacity mismatch should fail"),
        };
        assert!(matches!(
            raw_error,
            FimlError::FeatureVectorCapacityMismatch {
                expected: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn supports_empty_transformations_and_completely_empty_layouts() {
        let spec = ModelInputSpec::<f64>::with_capacity(
            FeatureVectorSpec::new([day_of_week("day")]).unwrap(),
            [],
            2,
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<f64, 1>::new(),
                ArrayFeatureVector::<f64, 2>::new_of_length(0),
            )
            .unwrap();
        pipeline.handle_event(Event::time(0)).unwrap();
        assert_eq!(pipeline.raw_values(), &[4.0]);
        assert!(pipeline.values().iter().all(|value| value.is_nan()));
        assert!(pipeline.output_ids().is_empty());

        let empty_raw = FeatureVectorSpec::new(Vec::<FeatureDefinition>::new()).unwrap();
        let empty_spec = ModelInputSpec::<f64>::new(empty_raw, []).unwrap();
        let mut empty_pipeline = empty_spec
            .build(
                ArrayFeatureVector::<f64, 0>::new(),
                ArrayFeatureVector::<f64, 0>::new(),
            )
            .unwrap();
        empty_pipeline.handle_event(Event::time(1)).unwrap();
        assert!(empty_pipeline.raw_values().is_empty());
        assert!(empty_pipeline.values().is_empty());
    }
}

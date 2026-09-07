//! Compiles raw feature extraction and scalar transformations into model input.
//!
//! The spec types validate named layouts on the cold path. [`Pipeline`] keeps
//! resolved indexes and preallocated transformer state, and writes one caller-owned
//! model vector directly on the event-processing hot path.

mod specs;

pub use specs::PipelineSpec;

use super::transformers::Transformer;
use crate::{Event, FeatureExtractor, FeatureId, FeatureVector, Result, UpdateResult};

/// Allocation-free event runtime for raw extraction and final model input.
pub struct Pipeline<RawV, ModelV>
where
    RawV: FeatureVector,
    ModelV: FeatureVector,
{
    feature_extractor: FeatureExtractor<RawV>,
    operations: Box<[Transformer]>,
    model_vector: ModelV,
    output_ids: Box<[FeatureId]>,
}

impl<RawV, ModelV> Pipeline<RawV, ModelV>
where
    RawV: FeatureVector,
    ModelV: FeatureVector,
{
    /// Applies an accepted event to raw features and then refreshes model input.
    /// Non-finite payloads are rejected before changing raw or final values,
    /// timed state, or the timestamp watermark, even for unsubscribed symbols.
    /// Finite zero and negative payloads are accepted.
    #[must_use = "event errors must be handled before using updated model-input values"]
    pub fn handle_event(&mut self, event: Event) -> Result<UpdateResult> {
        let update_result = self.feature_extractor.handle_event(event)?;
        let raw_values = self.feature_extractor.feature_vector().values();
        for operation in &mut self.operations {
            operation.apply(raw_values, &mut self.model_vector);
        }
        Ok(update_result)
    }

    /// Returns the raw extractor vector, including reserved cells.
    pub fn raw_values(&self) -> &[f64] {
        self.feature_extractor.feature_vector().values()
    }

    /// Returns final model input, including reserved cells.
    pub fn values(&self) -> &[f64] {
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
        ArrayFeatureVector, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureKey,
        FeatureSource, FimlError, InvalidArgumentError, InvalidTransformationDefinitionError,
        Symbol, TransformerDefinition, WarmupPolicy,
    };

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

    #[test]
    fn lagged_transformers_track_accepted_events_in_authored_output_order() {
        let raw_spec = FeatureExtractorSpec::new([time_since_first_event("elapsed")]).unwrap();
        let spec = PipelineSpec::with_capacity(
            raw_spec,
            [
                TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("lag2"), 2),
                TransformerDefinition::identity(FeatureId::new("elapsed"), FeatureId::new("now")),
                TransformerDefinition::standard_scale(
                    FeatureId::new("elapsed"),
                    FeatureId::new("scaled"),
                    0.0,
                    10.0,
                ),
                TransformerDefinition::lagged(FeatureId::new("elapsed"), FeatureId::new("lag1"), 1),
            ],
            5,
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<1>::new(),
                ArrayFeatureVector::<5>::new_of_length(4),
            )
            .unwrap();
        assert!(pipeline.values().iter().all(|value| value.is_nan()));
        assert_eq!(
            pipeline.output_ids(),
            &[
                FeatureId::new("lag2"),
                FeatureId::new("now"),
                FeatureId::new("scaled"),
                FeatureId::new("lag1"),
            ]
        );

        let cases = [
            (100, [f64::NAN, 0.0, 0.0, f64::NAN]),
            (110, [f64::NAN, 10.0, 1.0, 0.0]),
            (120, [0.0, 20.0, 2.0, 10.0]),
            (130, [10.0, 30.0, 3.0, 20.0]),
            (140, [20.0, 40.0, 4.0, 30.0]),
        ];
        for (timestamp, expected) in cases {
            if timestamp > 100 {
                let before = pipeline
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                assert!(pipeline.handle_event(Event::time(99)).is_err());
                assert!(
                    pipeline
                        .handle_event(Event::price(Symbol::GLOBAL, f64::NAN, timestamp))
                        .is_err()
                );
                assert_eq!(pipeline.last_timestamp(), Some(timestamp - 10));
                assert_eq!(
                    pipeline
                        .values()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    before
                );
            }
            pipeline.handle_event(Event::time(timestamp)).unwrap();
            assert_eq!(pipeline.raw_values(), &[(timestamp - 100) as f64]);
            for (actual, expected) in pipeline.values().iter().zip(expected) {
                assert!((actual.is_nan() && expected.is_nan()) || *actual == expected);
            }
            assert!(pipeline.values()[4].is_nan());
        }
    }

    #[test]
    fn rejects_zero_lag_before_compiling_a_transformer() {
        let error = PipelineSpec::new(
            FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
            [TransformerDefinition::lagged(
                FeatureId::new("day"),
                FeatureId::new("lagged"),
                0,
            )],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            FimlError::InvalidTransformationDefinition {
                index: 0,
                reason: InvalidTransformationDefinitionError::LagWindowZero,
            }
        ));
    }

    #[test]
    fn identity_and_standard_scaling_work() {
        let raw_spec = FeatureExtractorSpec::new([day_of_week("day")]).unwrap();
        let spec = PipelineSpec::new(
            raw_spec,
            [
                TransformerDefinition::identity(FeatureId::new("day"), FeatureId::new("identity")),
                TransformerDefinition::standard_scale(
                    FeatureId::new("day"),
                    FeatureId::new("scaled"),
                    2.0,
                    2.0,
                ),
            ],
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<1>::new(),
                ArrayFeatureVector::<2>::new(),
            )
            .unwrap();

        assert!(pipeline.values().iter().all(|value| value.is_nan()));
        pipeline.handle_event(Event::time(0)).unwrap();

        assert_eq!(pipeline.raw_values(), &[4.0]);
        assert_eq!(pipeline.values(), &[4.0, 1.0]);
    }

    #[test]
    fn authored_order_is_final_order_and_raw_and_final_ids_are_separate() {
        let raw_spec =
            FeatureExtractorSpec::new([time_since_first_event("elapsed"), day_of_week("day")])
                .unwrap();
        let spec = PipelineSpec::with_metadata(
            raw_spec,
            [
                TransformerDefinition::identity(
                    FeatureId::new("elapsed"),
                    FeatureId::new("elapsed"),
                ),
                TransformerDefinition::identity(FeatureId::new("day"), FeatureId::new("model_day")),
            ],
            4,
            Some("opaque".to_owned()),
        )
        .unwrap();

        assert_eq!(spec.feature_vector_length(), 2);
        assert_eq!(spec.feature_vector_capacity(), 4);
        assert_eq!(spec.checksum(), Some("opaque"));
        assert_eq!(spec.raw_feature_extractor_spec().feature_vector_length(), 2);
        assert_eq!(spec.transformation_definitions().len(), 2);

        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<2>::new(),
                ArrayFeatureVector::<4>::new_of_length(2),
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
            FeatureExtractorSpec::new([day_of_week("day"), time_since_first_event("elapsed")])
                .unwrap();
        let spec = PipelineSpec::new(
            raw_spec,
            [TransformerDefinition::standard_scale(
                FeatureId::new("day"),
                FeatureId::new("day"),
                2.0,
                2.0,
            )],
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<2>::new(),
                ArrayFeatureVector::<1>::new(),
            )
            .unwrap();

        pipeline.handle_event(Event::time(0)).unwrap();

        assert_eq!(pipeline.raw_values(), &[4.0, 0.0]);
        assert_eq!(pipeline.values(), &[1.0]);
        assert_eq!(pipeline.output_ids(), &[FeatureId::new("day")]);
    }

    #[test]
    fn warmup_nan_propagates_and_rejected_event_leaves_final_output_unchanged() {
        let raw_spec = FeatureExtractorSpec::new([warming_sma("sma")]).unwrap();
        let spec = PipelineSpec::new(
            raw_spec,
            [TransformerDefinition::standard_scale(
                FeatureId::new("sma"),
                FeatureId::new("scaled_sma"),
                10.0,
                5.0,
            )],
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<1>::new(),
                ArrayFeatureVector::<1>::new(),
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
        let unknown = PipelineSpec::new(
            FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
            [TransformerDefinition::identity(
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

        let duplicate = PipelineSpec::new(
            FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
            [
                TransformerDefinition::identity(FeatureId::new("day"), FeatureId::new("output")),
                TransformerDefinition::identity(FeatureId::new("day"), FeatureId::new("output")),
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

        let reserved = PipelineSpec::new(
            FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
            [TransformerDefinition::identity(
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

        let capacity = PipelineSpec::with_capacity(
            FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
            [TransformerDefinition::identity(
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
            let error = PipelineSpec::new(
                FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
                [TransformerDefinition::standard_scale(
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
            PipelineSpec::with_capacity(
                FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
                [TransformerDefinition::identity(
                    FeatureId::new("day"),
                    FeatureId::new("output"),
                )],
                2,
            )
            .unwrap()
        };

        let capacity_error = match make_spec().build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<1>::new(),
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
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<2>::new_of_length(0),
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
            ArrayFeatureVector::<2>::new_of_length(1),
            ArrayFeatureVector::<2>::new_of_length(1),
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
        let spec = PipelineSpec::with_capacity(
            FeatureExtractorSpec::new([day_of_week("day")]).unwrap(),
            [],
            2,
        )
        .unwrap();
        let mut pipeline = spec
            .build(
                ArrayFeatureVector::<1>::new(),
                ArrayFeatureVector::<2>::new_of_length(0),
            )
            .unwrap();
        pipeline.handle_event(Event::time(0)).unwrap();
        assert_eq!(pipeline.raw_values(), &[4.0]);
        assert!(pipeline.values().iter().all(|value| value.is_nan()));
        assert!(pipeline.output_ids().is_empty());

        let empty_raw = FeatureExtractorSpec::new(Vec::<FeatureDefinition>::new()).unwrap();
        let empty_spec = PipelineSpec::new(empty_raw, []).unwrap();
        let mut empty_pipeline = empty_spec
            .build(
                ArrayFeatureVector::<0>::new(),
                ArrayFeatureVector::<0>::new(),
            )
            .unwrap();
        empty_pipeline.handle_event(Event::time(1)).unwrap();
        assert!(empty_pipeline.raw_values().is_empty());
        assert!(empty_pipeline.values().is_empty());
    }
}

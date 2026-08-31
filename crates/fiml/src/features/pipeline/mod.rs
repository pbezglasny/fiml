//! Compiles raw feature extraction and scalar transformations into model input.
//!
//! The spec types validate named layouts on the cold path. [`Pipeline`] keeps
//! only resolved indexes and writes one caller-owned model vector directly on
//! the event-processing hot path.

use crate::{
    Event, FeatureExtractor, FeatureId, FeatureVector, FeatureVectorSpec, FimlError, Float,
    InvalidArgumentError, InvalidTransformationDefinitionError, Result, UpdateResult,
};

/// One named scalar transformation from the raw feature layout to model input.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformationDefinition<F> {
    /// Copies one raw scalar without changing its value.
    Identity { input: FeatureId, output: FeatureId },
    /// Applies `(input - mean) / scale` to one raw scalar.
    StandardScale {
        input: FeatureId,
        output: FeatureId,
        mean: F,
        scale: F,
    },
}

impl<F> TransformationDefinition<F> {
    /// Creates a scalar identity transformation.
    pub fn identity(input: FeatureId, output: FeatureId) -> Self {
        Self::Identity { input, output }
    }

    /// Creates a scalar standard-scaling transformation.
    pub fn standard_scale(input: FeatureId, output: FeatureId, mean: F, scale: F) -> Self {
        Self::StandardScale {
            input,
            output,
            mean,
            scale,
        }
    }

    fn input(&self) -> &FeatureId {
        match self {
            Self::Identity { input, .. } | Self::StandardScale { input, .. } => input,
        }
    }

    fn output(&self) -> &FeatureId {
        match self {
            Self::Identity { output, .. } | Self::StandardScale { output, .. } => output,
        }
    }
}

/// Validated configuration for raw extraction and the final model-input layout.
///
/// Transformations remain in authored order, which is also final vector order.
/// Raw and final IDs occupy separate layouts and may therefore use the same name.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInputSpec<F> {
    raw_feature_vector_spec: FeatureVectorSpec,
    transformation_definitions: Vec<TransformationDefinition<F>>,
    feature_vector_capacity: usize,
    checksum: Option<String>,
}

impl<F: Float> ModelInputSpec<F> {
    /// Creates a spec whose final width equals its transformation count.
    pub fn new(
        raw_feature_vector_spec: FeatureVectorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformationDefinition<F>>,
    ) -> Result<Self> {
        let transformation_definitions = transformation_definitions.into_iter().collect::<Vec<_>>();
        let capacity = transformation_definitions.len();
        Self::with_metadata(
            raw_feature_vector_spec,
            transformation_definitions,
            capacity,
            None,
        )
    }

    /// Creates a spec with explicit final width and trailing reserved cells.
    pub fn with_capacity(
        raw_feature_vector_spec: FeatureVectorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformationDefinition<F>>,
        feature_vector_capacity: usize,
    ) -> Result<Self> {
        Self::with_metadata(
            raw_feature_vector_spec,
            transformation_definitions,
            feature_vector_capacity,
            None,
        )
    }

    /// Creates a spec with explicit final width and opaque checksum metadata.
    pub fn with_metadata(
        raw_feature_vector_spec: FeatureVectorSpec,
        transformation_definitions: impl IntoIterator<Item = TransformationDefinition<F>>,
        feature_vector_capacity: usize,
        checksum: Option<String>,
    ) -> Result<Self> {
        let transformation_definitions = transformation_definitions.into_iter().collect::<Vec<_>>();
        if feature_vector_capacity < transformation_definitions.len() {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::FeatureVectorCapacityTooSmall {
                    capacity: feature_vector_capacity,
                    active_length: transformation_definitions.len(),
                },
            ));
        }

        for (index, definition) in transformation_definitions.iter().enumerate() {
            if !raw_feature_vector_spec
                .definitions()
                .iter()
                .any(|raw| raw.id == *definition.input())
            {
                return invalid_definition(
                    index,
                    InvalidTransformationDefinitionError::InputFeatureNotFound,
                );
            }
            if crate::features::is_reserved_feature_id(definition.output()) {
                return invalid_definition(
                    index,
                    InvalidTransformationDefinitionError::ReservedOutputFeature,
                );
            }
            if transformation_definitions[..index]
                .iter()
                .any(|previous| previous.output() == definition.output())
            {
                return invalid_definition(
                    index,
                    InvalidTransformationDefinitionError::DuplicateOutputFeature,
                );
            }
            if let TransformationDefinition::StandardScale { mean, scale, .. } = definition {
                if !mean.is_finite() {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::MeanNotFinite,
                    );
                }
                if !scale.is_finite() {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::ScaleNotFinite,
                    );
                }
                if *scale <= F::ZERO {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::ScaleNotPositive,
                    );
                }
                if !(F::ONE / *scale).is_finite() {
                    return invalid_definition(
                        index,
                        InvalidTransformationDefinitionError::InverseScaleNotFinite,
                    );
                }
            }
        }

        Ok(Self {
            raw_feature_vector_spec,
            transformation_definitions,
            feature_vector_capacity,
            checksum,
        })
    }

    /// Returns the raw feature-extraction configuration.
    pub fn raw_feature_vector_spec(&self) -> &FeatureVectorSpec {
        &self.raw_feature_vector_spec
    }

    /// Returns scalar transformations in final model-vector order.
    pub fn transformation_definitions(&self) -> &[TransformationDefinition<F>] {
        &self.transformation_definitions
    }

    /// Returns the complete final width, including reserved cells.
    pub fn feature_vector_capacity(&self) -> usize {
        self.feature_vector_capacity
    }

    /// Returns the number of active final outputs.
    pub fn feature_vector_length(&self) -> usize {
        self.transformation_definitions.len()
    }

    /// Returns opaque checksum metadata without interpreting or verifying it.
    pub fn checksum(&self) -> Option<&str> {
        self.checksum.as_deref()
    }

    /// Compiles this spec into caller-supplied raw and final storage.
    pub fn build<RawV, ModelV>(
        &self,
        raw_vector: RawV,
        mut model_vector: ModelV,
    ) -> Result<Pipeline<F, RawV, ModelV>>
    where
        RawV: FeatureVector<F = F>,
        ModelV: FeatureVector<F = F>,
    {
        if model_vector.capacity() != self.feature_vector_capacity {
            return Err(FimlError::ModelVectorCapacityMismatch {
                expected: self.feature_vector_capacity,
                actual: model_vector.capacity(),
            });
        }
        if model_vector.len() != self.transformation_definitions.len() {
            return Err(FimlError::ModelVectorLengthMismatch {
                expected: self.transformation_definitions.len(),
                actual: model_vector.len(),
            });
        }

        let feature_extractor = self.raw_feature_vector_spec.build(raw_vector)?;
        let mut operations = Vec::with_capacity(self.transformation_definitions.len());
        let mut output_ids = Vec::with_capacity(self.transformation_definitions.len());
        for (output_index, definition) in self.transformation_definitions.iter().enumerate() {
            let input_index = feature_extractor
                .feature_index(definition.input())
                .expect("model-input construction validated every raw input ID");
            let operation = match definition {
                TransformationDefinition::Identity { .. } => ScalarOperation::Identity {
                    input_index,
                    output_index,
                },
                TransformationDefinition::StandardScale { mean, scale, .. } => {
                    ScalarOperation::StandardScale {
                        input_index,
                        output_index,
                        mean: *mean,
                        inverse_scale: F::ONE / *scale,
                    }
                }
            };
            operations.push(operation);
            output_ids.push(definition.output().clone());
        }
        for index in 0..model_vector.capacity() {
            model_vector.set_value_at(index, F::NAN);
        }

        Ok(Pipeline {
            feature_extractor,
            operations: operations.into_boxed_slice(),
            model_vector,
            output_ids: output_ids.into_boxed_slice(),
        })
    }
}

fn invalid_definition<T>(index: usize, reason: InvalidTransformationDefinitionError) -> Result<T> {
    Err(FimlError::InvalidTransformationDefinition { index, reason })
}

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
        ArrayFeatureVector, EventField, FeatureDefinition, FeatureKey, FeatureSource, Symbol,
        WarmupPolicy,
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

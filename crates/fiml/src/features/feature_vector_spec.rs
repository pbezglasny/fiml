use crate::{FeatureDefinition, FeatureExtractor, FeatureSource, FeatureVector, FimlError, Float};

use super::FeatureKey;

const RESERVED_ID_PREFIX: &str = "__reserved_";

/// Versioned configuration for one complete model-input feature vector.
///
/// The spec owns canonically ordered scalar definitions plus the model width.
/// Definitions describe active outputs; any remaining capacity is reserved and
/// is initialized to `NaN` when the spec is compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureVectorSpec {
    definitions: Vec<FeatureDefinition>,
    feature_vector_capacity: usize,
    checksum: Option<String>,
}

impl FeatureVectorSpec {
    /// Creates a spec whose model width exactly matches its active output count.
    pub fn new(
        definitions: impl IntoIterator<Item = FeatureDefinition>,
    ) -> Result<Self, FimlError> {
        let definitions = definitions.into_iter().collect::<Vec<_>>();
        let capacity = definitions.len();
        Self::with_metadata(definitions, capacity, None)
    }

    /// Creates a spec with an explicit model width, reserving trailing cells.
    pub fn with_capacity(
        definitions: impl IntoIterator<Item = FeatureDefinition>,
        feature_vector_capacity: usize,
    ) -> Result<Self, FimlError> {
        Self::with_metadata(definitions, feature_vector_capacity, None)
    }

    /// Creates a spec with an explicit model width and opaque checksum metadata.
    pub fn with_metadata(
        definitions: impl IntoIterator<Item = FeatureDefinition>,
        feature_vector_capacity: usize,
        checksum: Option<String>,
    ) -> Result<Self, FimlError> {
        let mut definitions = definitions.into_iter().collect::<Vec<_>>();
        if feature_vector_capacity < definitions.len() {
            return Err(FimlError::InvalidArgument(format!(
                "feature vector capacity {feature_vector_capacity} is smaller than active length {}",
                definitions.len()
            )));
        }
        if let Some(definition) = definitions
            .iter()
            .find(|definition| definition.id.as_str().starts_with(RESERVED_ID_PREFIX))
        {
            return Err(FimlError::InvalidArgument(format!(
                "feature ID {:?} uses the reserved namespace {RESERVED_ID_PREFIX:?}",
                definition.id.as_str()
            )));
        }
        definitions.sort_by_cached_key(|definition| canonical_sort_key(&definition.key));
        Ok(Self {
            definitions,
            feature_vector_capacity,
            checksum,
        })
    }

    /// Returns active scalar definitions in canonical model-input order.
    pub fn definitions(&self) -> &[FeatureDefinition] {
        &self.definitions
    }

    /// Returns the complete model-input width, including reserved cells.
    pub fn feature_vector_capacity(&self) -> usize {
        self.feature_vector_capacity
    }

    /// Returns the number of active scalar outputs.
    pub fn feature_vector_length(&self) -> usize {
        self.definitions.len()
    }

    /// Returns the number of canonical grouped indicator configurations.
    pub fn indicator_count(&self) -> usize {
        self.definitions
            .iter()
            .map(|definition| canonical_sort_key(&definition.key))
            .fold((0, None), |(count, previous), key| {
                if previous.as_ref() == Some(&key) {
                    (count, previous)
                } else {
                    (count + 1, Some(key))
                }
            })
            .0
    }

    /// Returns opaque checksum metadata without interpreting or verifying it.
    pub fn checksum(&self) -> Option<&str> {
        self.checksum.as_deref()
    }

    /// Compiles this configuration into the supplied output storage.
    ///
    /// Storage capacity must equal the configured model width and its active
    /// length must equal the number of definitions. Every cell is reset to
    /// `NaN`, including trailing reserved cells, before compilation.
    pub fn build<F, V>(&self, mut output_vector: V) -> Result<FeatureExtractor<F, V>, FimlError>
    where
        F: Float,
        V: FeatureVector<F = F>,
    {
        if output_vector.capacity() != self.feature_vector_capacity {
            return Err(FimlError::FeatureVectorCapacityMismatch {
                expected: self.feature_vector_capacity,
                actual: output_vector.capacity(),
            });
        }
        if output_vector.len() != self.definitions.len() {
            return Err(FimlError::OutputCountMismatch {
                expected: self.definitions.len(),
                actual: output_vector.len(),
            });
        }
        for index in 0..output_vector.capacity() {
            output_vector.set_value_at(index, F::NAN);
        }

        let compilation = super::compiler::compile(self.definitions.clone(), output_vector.len())?;
        Ok(FeatureExtractor::new(output_vector, compilation))
    }
}

fn canonical_sort_key(key: &FeatureKey) -> (bool, String, u8, u8, u8, u128, u128, i64) {
    let (symbol, kind, source, warmup, aggregation, scalar_identity, utc_offset) = match key {
        FeatureKey::Sma {
            symbol,
            source,
            warmup_policy,
            ..
        } => (*symbol, 4, *source, warmup_rank(*warmup_policy), 0, 0, 0),
        FeatureKey::Ema {
            symbol,
            source,
            warmup_policy,
            ..
        } => (*symbol, 2, *source, warmup_rank(*warmup_policy), 0, 0, 0),
        FeatureKey::Cvd {
            symbol,
            source,
            warmup_policy,
            ..
        } => (*symbol, 0, *source, warmup_rank(*warmup_policy), 0, 0, 0),
        FeatureKey::SmaTimed {
            symbol,
            source,
            aggregation,
            warmup_policy,
            ..
        } => (
            *symbol,
            5,
            *source,
            warmup_rank(*warmup_policy),
            aggregation.as_nanos(),
            0,
            0,
        ),
        FeatureKey::ObvTimed {
            symbol,
            source,
            aggregation,
            warmup_policy,
            ..
        } => (
            *symbol,
            3,
            *source,
            warmup_rank(*warmup_policy),
            aggregation.as_nanos(),
            0,
            0,
        ),
        FeatureKey::TradeCountTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => (
            *symbol,
            7,
            *source,
            warmup_rank(*warmup_policy),
            aggregation.as_nanos(),
            window.as_nanos(),
            0,
        ),
        FeatureKey::DayOfWeek { symbol, source } => (*symbol, 1, *source, 0, 0, 0, 0),
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol,
            source,
            utc_offset_millis,
        } => (*symbol, 6, *source, 0, 0, 0, *utc_offset_millis),
    };
    (
        symbol != crate::Symbol::GLOBAL,
        symbol.resolve_as_string(),
        kind,
        source_rank(source),
        warmup,
        aggregation,
        scalar_identity,
        utc_offset,
    )
}

const fn source_rank(source: FeatureSource) -> u8 {
    match source {
        FeatureSource::Field(crate::EventField::Price) => 0,
        FeatureSource::Field(crate::EventField::TradePrice) => 1,
        FeatureSource::Field(crate::EventField::TradeVolume) => 2,
        FeatureSource::Field(crate::EventField::Volume) => 3,
        FeatureSource::Event(kind) => 16 + kind as u8,
        FeatureSource::EveryEvent => u8::MAX,
    }
}

const fn warmup_rank(warmup: crate::WarmupPolicy) -> u8 {
    match warmup {
        crate::WarmupPolicy::FirstValue => 0,
        crate::WarmupPolicy::FullWindow => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, Event, FeatureId, FeatureVector, Symbol};

    fn day_of_week(id: &str) -> FeatureDefinition {
        FeatureDefinition::new(
            FeatureKey::DayOfWeek {
                symbol: Symbol::GLOBAL,
                source: FeatureSource::EveryEvent,
            },
            FeatureId::new(id),
        )
    }

    #[test]
    fn build_initializes_active_and_reserved_cells_to_nan() {
        let spec = FeatureVectorSpec::with_capacity([day_of_week("day")], 3).unwrap();
        let mut output = ArrayFeatureVector::<f64, 3>::new_of_length(1);
        output.set_value_at(0, 1.0);
        output.set_value_at(1, 2.0);
        output.set_value_at(2, 3.0);

        let mut extractor = spec.build(output).unwrap();
        assert!(
            extractor
                .feature_vector()
                .values()
                .iter()
                .all(|value| value.is_nan())
        );

        extractor.handle_event(&Event::time(0)).unwrap();
        assert!(!extractor.feature_vector().values()[0].is_nan());
        assert!(
            extractor.feature_vector().values()[1..]
                .iter()
                .all(|value| value.is_nan())
        );
        assert_eq!(extractor.feature_ids(), &[FeatureId::new("day")]);
    }

    #[test]
    fn build_rejects_capacity_and_active_length_mismatches() {
        let spec = FeatureVectorSpec::with_capacity([day_of_week("day")], 3).unwrap();
        let capacity_error =
            match spec.build::<f64, _>(ArrayFeatureVector::<f64, 2>::new_of_length(1)) {
                Err(error) => error,
                Ok(_) => panic!("capacity mismatch should fail"),
            };
        assert!(matches!(
            capacity_error,
            FimlError::FeatureVectorCapacityMismatch {
                expected: 3,
                actual: 2
            }
        ));

        let length_error =
            match spec.build::<f64, _>(ArrayFeatureVector::<f64, 3>::new_of_length(2)) {
                Err(error) => error,
                Ok(_) => panic!("active length mismatch should fail"),
            };
        assert!(matches!(
            length_error,
            FimlError::OutputCountMismatch {
                expected: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn reserved_feature_id_namespace_is_rejected() {
        let error = FeatureVectorSpec::new([day_of_week("__reserved_0")]).unwrap_err();
        assert!(error.to_string().contains("reserved namespace"));
    }
}

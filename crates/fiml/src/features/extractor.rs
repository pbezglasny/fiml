//! Runtime-sized extractor compiled from a grouped [`FeatureSet`].

use std::fmt;

use crate::features::FeatureSet;
use crate::features::event::Event;
use crate::features::indicator_vector::{IndicatorFeatureVector, IndicatorFeatures};
use crate::vectors::{FeatureVector, VecFeatureVector};
use crate::{FimlError, Result};

/// Location and cause of the first invalid event in a proposed dispatch batch.
#[derive(Debug)]
pub struct DispatchSequenceError {
    pub index: usize,
    pub error: FimlError,
}

impl fmt::Display for DispatchSequenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "event {}: {}", self.index, self.error)
    }
}

macro_rules! dynamic_extractor {
    ($($variant:ident => $capacity:literal),+ $(,)?) => {
        pub enum FeatureExtractor {
            $(
                $variant(Box<IndicatorFeatureVector<f64, VecFeatureVector<f64>, $capacity>>),
            )+
        }

        impl FeatureExtractor {
            pub const MAX_INDICATORS: usize = {
                let mut max = 0;
                $(if $capacity > max { max = $capacity; })+
                max
            };

            pub const MAX_OUTPUTS: usize = 128;

            fn build_sized(feature_set: &FeatureSet) -> Result<Self> {
                let indicator_count = feature_set.indicator_count();
                let output_count = feature_set.output_count();
                if output_count > Self::MAX_OUTPUTS {
                    return Err(FimlError::TooManyOutputs {
                        count: output_count,
                        capacity: Self::MAX_OUTPUTS,
                    });
                }

                $(
                    if indicator_count <= $capacity {
                        let mut cells = VecFeatureVector::new(output_count);
                        for index in 0..output_count {
                            cells.set_value_at(index, f64::NAN);
                        }
                        return Ok(Self::$variant(Box::new(
                            IndicatorFeatureVector::from_feature_set(cells, feature_set)?,
                        )));
                    }
                )+

                Err(FimlError::TooManyIndicators {
                    count: indicator_count,
                    capacity: Self::MAX_INDICATORS,
                })
            }

            pub fn feature_names(&self) -> &[String] {
                match self {
                    $(Self::$variant(extractor) => extractor.feature_names(),)+
                }
            }

            pub fn last_timestamp(&self) -> Option<i64> {
                match self {
                    $(Self::$variant(extractor) => extractor.last_timestamp(),)+
                }
            }
        }

        impl IndicatorFeatures for FeatureExtractor {
            type F = f64;
            type FeatureVector = VecFeatureVector<f64>;

            fn feature_vector(&self) -> &Self::FeatureVector {
                match self {
                    $(Self::$variant(extractor) => extractor.feature_vector(),)+
                }
            }

            fn dispatch(&mut self, event: &Event<f64>) -> Result<()> {
                match self {
                    $(Self::$variant(extractor) => extractor.dispatch(event),)+
                }
            }

            fn validate_dispatch(&self, event: &Event<f64>) -> Result<()> {
                match self {
                    $(Self::$variant(extractor) => extractor.validate_dispatch(event),)+
                }
            }

            fn index_of(&self, canonical_name: &str) -> Option<usize> {
                match self {
                    $(Self::$variant(extractor) => extractor.index_of(canonical_name),)+
                }
            }
        }
    };
}

dynamic_extractor! {
    Cap16 => 16,
    Cap32 => 32,
    Cap48 => 48,
    Cap64 => 64,
    Cap80 => 80,
    Cap96 => 96,
    Cap112 => 112,
    Cap128 => 128,
}

impl FeatureExtractor {
    pub fn from_feature_set(feature_set: &FeatureSet) -> Result<Self> {
        Self::build_sized(feature_set)
    }

    pub fn values(&self) -> &[f64] {
        self.feature_vector().values()
    }

    pub fn has_dispatched(&self) -> bool {
        self.last_timestamp().is_some()
    }

    /// Validate a complete batch against the extractor's global watermark
    /// without mutating indicator state.
    pub fn validate_dispatch_sequence(
        &self,
        events: &[Event<f64>],
    ) -> std::result::Result<(), DispatchSequenceError> {
        let mut previous_timestamp = self.last_timestamp();
        for (index, event) in events.iter().enumerate() {
            if let Some(previous_timestamp) = previous_timestamp
                && event.timestamp() < previous_timestamp
            {
                return Err(DispatchSequenceError {
                    index,
                    error: FimlError::TimestampOutOfOrder {
                        symbol: event.symbol(),
                        event_kind: event.kind(),
                        timestamp: event.timestamp(),
                        previous_timestamp,
                    },
                });
            }
            previous_timestamp = Some(event.timestamp());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::features::{IndicatorDef, IndicatorSpec, ValueSource};
    use crate::{Event, IndicatorFeatures, symbols};

    use super::*;

    fn feature_set() -> FeatureSet {
        FeatureSet::new(vec![IndicatorDef::symbol(
            "AAPL",
            IndicatorSpec::Sma {
                source: ValueSource::Price,
                windows: vec![2, 5],
            },
        )])
    }

    #[test]
    fn runtime_capacity_counts_indicators_separately_from_outputs() {
        let extractor = FeatureExtractor::from_feature_set(&feature_set()).unwrap();

        assert!(matches!(extractor, FeatureExtractor::Cap16(_)));
        assert_eq!(extractor.values().len(), 2);
        assert_eq!(extractor.feature_names().len(), 2);
    }

    #[test]
    fn generated_names_and_values_are_in_window_order() {
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set()).unwrap();
        let aapl = symbols::intern("AAPL");
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            extractor.dispatch(&Event::price(aapl, value, 0)).unwrap();
        }

        assert_eq!(
            extractor.feature_names(),
            ["AAPL:price:sma:2", "AAPL:price:sma:5"]
        );
        assert_eq!(extractor.index_of("AAPL:price:sma:5"), Some(1));
        assert_eq!(extractor.values(), [4.5, 3.0]);
    }

    #[test]
    fn batch_validation_uses_global_watermark_without_mutation() {
        let mut extractor = FeatureExtractor::from_feature_set(&feature_set()).unwrap();
        extractor.dispatch(&Event::time(100)).unwrap();
        let events = [Event::time(101), Event::time(99)];

        let error = extractor.validate_dispatch_sequence(&events).unwrap_err();

        assert_eq!(error.index, 1);
        assert_eq!(extractor.last_timestamp(), Some(100));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_round_trip_preserves_grouping_and_output() {
        let feature_set = feature_set();
        let json = serde_json::to_string(&feature_set).unwrap();
        let restored: FeatureSet = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, feature_set);
        assert_eq!(restored.indicator_count(), 1);
        assert_eq!(restored.output_count(), 2);
    }
}

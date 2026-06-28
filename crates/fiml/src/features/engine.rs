//! Runtime-sized engine built from a [`EngineSpec`].
//!
//! [`IndicatorFeatureVector`] fixes its feature-array capacity `M` at compile
//! time. When the feature set comes from a deserialized spec that count is only
//! known at runtime, so [`DynIndicatorEngine`] wraps a fixed set of capacity
//! variants (16, 32, ... 128) and [`from_spec`](DynIndicatorEngine::from_spec)
//! picks the smallest one that fits. The cell storage is the heap-backed
//! [`VecFeatureVector`], sized to the exact number of features.
//!
//! Each variant is boxed: the wrapped vector embeds a `[_; M]` feature array
//! that is large for big `M`, so keeping it behind a pointer keeps the enum
//! itself pointer-sized and selecting the smallest capacity minimizes the
//! single heap allocation. Building an engine is a cold path; `dispatch` is not
//! affected.

use crate::features::builtin::BuiltinFeature;
use crate::features::event::Event;
use crate::features::indicator_vector::{IndicatorFeatureVector, IndicatorFeatures};
use crate::features::spec::EngineSpec;
use crate::vectors::{FeatureVector, VecFeatureVector};
use crate::{BuiltinSpec, FimlError, Result, Symbol, symbols};

/// Generate the capacity-variant enum and forward every operation to the
/// wrapped [`IndicatorFeatureVector`]. Each variant pairs an identifier with the
/// compile-time capacity it carries.
macro_rules! dyn_indicator_engine {
    ($($variant:ident => $cap:literal),+ $(,)?) => {
        /// Engine whose feature-array capacity is selected at runtime from a
        /// fixed set of compile-time sizes. All variants share the same output
        /// storage type ([`VecFeatureVector<f64>`]) and the same `f64` element
        /// type, so the engine exposes a single [`IndicatorFeatures`] interface.
        pub enum DynIndicatorEngine {
            $($variant(Box<IndicatorFeatureVector<f64, VecFeatureVector<f64>, BuiltinFeature<f64>, $cap>>),)+
        }

        impl DynIndicatorEngine {
            /// Largest feature count any variant can hold.
            pub const MAX_FEATURES: usize = {
                let mut max = 0;
                $(if $cap > max { max = $cap; })+
                max
            };

            fn build_sized(specs: &[(&str, Symbol, BuiltinSpec)]) -> Result<Self> {
                let count = specs.len();
                $(
                    if count <= $cap {
                        let cells = VecFeatureVector::new(count);
                        return Ok(Self::$variant(Box::new(
                            IndicatorFeatureVector::from_builtin_specs(cells, specs)?,
                        )));
                    }
                )+
                Err(FimlError::InvalidArgument(format!(
                    "too many features: {count} (max {})",
                    Self::MAX_FEATURES
                )))
            }

            /// Feature names in output-cell order (see
            /// [`IndicatorFeatureVector::feature_names`]).
            pub fn feature_names(&self) -> Vec<String> {
                match self {
                    $(Self::$variant(engine) => engine.feature_names(),)+
                }
            }
        }

        impl IndicatorFeatures for DynIndicatorEngine {
            type F = f64;
            type FeatureVector = VecFeatureVector<f64>;

            fn feature_vector(&self) -> &Self::FeatureVector {
                match self {
                    $(Self::$variant(engine) => IndicatorFeatures::feature_vector(engine.as_ref()),)+
                }
            }

            fn dispatch(&mut self, event: &Event<f64>) {
                match self {
                    $(Self::$variant(engine) => engine.dispatch(event),)+
                }
            }

            fn index_of(&self, symbol: Symbol, name: &str) -> Option<usize> {
                match self {
                    $(Self::$variant(engine) => engine.index_of(symbol, name),)+
                }
            }
        }
    };
}

dyn_indicator_engine! {
    Cap16 => 16,
    Cap32 => 32,
    Cap48 => 48,
    Cap64 => 64,
    Cap80 => 80,
    Cap96 => 96,
    Cap112 => 112,
    Cap128 => 128,
}

impl DynIndicatorEngine {
    /// Build a runnable engine from a declarative [`EngineSpec`].
    ///
    /// Symbol names are interned in spec order, the smallest capacity variant
    /// that fits the feature count is selected, and each feature is wired to its
    /// output cell. Returns an error if there are more features than
    /// [`MAX_FEATURES`](Self::MAX_FEATURES) or any spec is invalid.
    pub fn from_spec(spec: &EngineSpec) -> Result<Self> {
        let specs: Vec<(&str, Symbol, BuiltinSpec)> = spec
            .features
            .iter()
            .map(|feature| {
                (
                    feature.name.as_str(),
                    symbols::intern(&feature.symbol),
                    feature.spec.clone(),
                )
            })
            .collect();
        Self::build_sized(&specs)
    }

    /// Current feature values in output-cell order.
    pub fn values(&self) -> &[f64] {
        IndicatorFeatures::feature_vector(self).values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::spec::FeatureSpec;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn sma(name: &str, symbol: &str, period: usize) -> FeatureSpec {
        FeatureSpec {
            name: name.to_string(),
            symbol: symbol.to_string(),
            spec: BuiltinSpec::Sma { period },
        }
    }

    #[test]
    fn from_spec_picks_smallest_capacity_variant() {
        let spec = EngineSpec::new(vec![sma("sma_2", "AAPL", 2)]);

        let engine = DynIndicatorEngine::from_spec(&spec).unwrap();

        assert!(matches!(engine, DynIndicatorEngine::Cap16(_)));
        assert_eq!(engine.values().len(), 1);
    }

    #[test]
    fn from_spec_builds_runnable_engine() {
        let spec = EngineSpec::new(vec![sma("sma_2", "AAPL", 2), sma("sma_5", "AAPL", 5)]);
        let aapl = symbols::intern("AAPL");
        let mut engine = DynIndicatorEngine::from_spec(&spec).unwrap();

        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            engine.dispatch(&Event::price(aapl, value, 0));
        }

        assert_eq!(engine.feature_names(), vec!["sma_2", "sma_5"]);
        // sma_2: mean(4, 5) = 4.5 ; sma_5: mean(1..=5) = 3.0
        assert!(approx_eq(engine.values()[0], 4.5));
        assert!(approx_eq(engine.values()[1], 3.0));
        assert_eq!(engine.index_of(aapl, "sma_5"), Some(1));
    }

    #[test]
    fn from_spec_rejects_more_features_than_max() {
        let features = (0..=DynIndicatorEngine::MAX_FEATURES)
            .map(|i| sma(&format!("sma_{i}"), "AAPL", 2))
            .collect();
        let spec = EngineSpec::new(features);

        assert!(DynIndicatorEngine::from_spec(&spec).is_err());
    }

    /// The parity contract: a spec round-tripped through JSON must rebuild an
    /// engine that produces exactly the same output as one built from the spec
    /// directly. This is what lets Python (batch) and Rust (live) agree.
    #[cfg(feature = "serde")]
    #[test]
    fn json_round_trip_preserves_exact_output() {
        let spec = EngineSpec::new(vec![
            sma("sma_2", "AAPL", 2),
            FeatureSpec {
                name: "ema_3".to_string(),
                symbol: "AAPL".to_string(),
                spec: BuiltinSpec::Ema { period: 3 },
            },
        ]);
        let aapl = symbols::intern("AAPL");

        let json = serde_json::to_string(&spec).unwrap();
        let restored: EngineSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, spec);

        let mut direct = DynIndicatorEngine::from_spec(&spec).unwrap();
        let mut from_json = DynIndicatorEngine::from_spec(&restored).unwrap();

        for value in [10.0, 11.0, 9.0, 12.0, 13.0] {
            direct.dispatch(&Event::price(aapl, value, 0));
            from_json.dispatch(&Event::price(aapl, value, 0));
        }

        // Exact equality, not approximate: identical spec + identical code path.
        assert_eq!(direct.values(), from_json.values());
        assert_eq!(direct.feature_names(), from_json.feature_names());
    }
}

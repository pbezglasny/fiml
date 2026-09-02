//! Owns versioned feature and model-input specifications and Serde adapters.
//!
//! The module's intent is to keep layout metadata and wire-format rules at one
//! seam. The specification types remain available without optional Cargo
//! features; enabling `serde` adds canonical serialization and strict
//! deserialization without changing their public interfaces.

mod feature_vector_spec;
#[cfg(feature = "serde")]
mod model_input_spec;
#[cfg(feature = "serde")]
mod serialization;

pub use feature_vector_spec::FeatureVectorSpec;

//! Owns the versioned feature-vector specification and its Serde adapter.
//!
//! The module's intent is to keep model-layout metadata and wire-format rules
//! at one seam. [`FeatureVectorSpec`] remains available without optional Cargo
//! features; enabling the `serde` feature adds serialization and deserialization
//! without changing the spec's public interface.

mod feature_vector_spec;
#[cfg(feature = "serde")]
mod serialization;

pub use feature_vector_spec::FeatureVectorSpec;

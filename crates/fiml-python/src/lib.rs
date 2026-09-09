//! Python bindings for the `fiml` feature extractor.
//!
//! The bindings deliberately run the *exact* Rust extractor: features are
//! computed by replaying events through [`fiml::FeatureExtractor`]'s dispatch,
//! the same code the live Rust environment uses. Feed both sides the same
//! feature-vector spec and events in the same order to get identical output. Indicator
//! state is always `f64`; Python arrays can be returned as `float32` or `float64`.

mod feature_extractor;
mod feature_extractor_spec;
mod model_input_pipeline;
mod order_book;
mod pipeline_spec;
mod runtime;

use fiml::{Symbol, symbols};
use pyo3::{exceptions::PyValueError, prelude::*};

use feature_extractor::FeatureExtractor;
use feature_extractor_spec::{FeatureExtractorSpec, PyWarmupPolicy};
use model_input_pipeline::ModelInputPipeline;
use order_book::OrderBookEvent;
use pipeline_spec::PipelineSpec;
use runtime::{
    KIND_ORDERBOOK, KIND_PRICE, KIND_TIME, KIND_TRADE, KIND_VOLUME, SIDE_AGGRESSOR_BUY,
    SIDE_AGGRESSOR_SELL,
};

/// Interns a symbol and translates core errors for all Python callers.
pub(crate) fn intern_symbol(name: &str) -> PyResult<Symbol> {
    symbols::intern(name).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn _fiml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OrderBookEvent>()?;
    m.add_class::<PyWarmupPolicy>()?;
    m.add_class::<FeatureExtractorSpec>()?;
    m.add_class::<PipelineSpec>()?;
    m.add_class::<FeatureExtractor>()?;
    m.add_class::<ModelInputPipeline>()?;
    m.add("KIND_PRICE", KIND_PRICE)?;
    m.add("KIND_VOLUME", KIND_VOLUME)?;
    m.add("KIND_TRADE", KIND_TRADE)?;
    m.add("KIND_ORDERBOOK", KIND_ORDERBOOK)?;
    m.add("KIND_TIME", KIND_TIME)?;
    m.add("SIDE_AGGRESSOR_BUY", SIDE_AGGRESSOR_BUY)?;
    m.add("SIDE_AGGRESSOR_SELL", SIDE_AGGRESSOR_SELL)?;
    Ok(())
}

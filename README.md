Fiml
======

Description
---------

Fiml is a feature engineering library for trading and machine learning.
Build features from historical market data in Python, then use the same
calculations on live events in Rust. Its feature vectors can serve as inputs
to models built with LightGBM, XGBoost, CatBoost, scikit-learn, and other ML
libraries.

* **Streaming computation** - Update indicators and feature vectors as price,
  volume, trade, and order-book events arrive, without recalculating the entire
  history.
* **One calculation engine** - Python bindings run the Rust engine, keeping
  historical feature generation and live computation consistent for the same
  configuration and event sequence.
* **Minimal allocation** - Reuse output storage and borrow feature values
  directly in Rust without copying a vector. Construction allocates internal
  state; allocation costs depend on the operation and API used.
* **Portable pipelines** - Export feature definitions and fitted transformation
  parameters from Python as JSON, then rebuild the feature pipeline in Rust.
* **Model preprocessing** - Combine indicators with transformations such as
  scaling, lagged values, missing-value imputation, and PCA to produce the
  model's input vector.

Fiml is under active development. Public APIs may change before the first
release.

Overview
------

Fiml turns a stream of market events into an ordered vector of model inputs:

```text
Events → Feature extractor → Raw features → Transformations → Model inputs
```

* **Events** carry timestamped price, volume, trade, or order-book updates.
  Time events support global calendar features. The same event stream can
  come from historical data or a live feed.
* **Feature definitions** describe each output: the calculation, its symbol
  and source, parameters such as window size, and an output ID.
  `FeatureExtractorSpec` collects these definitions into a reusable configuration.
* **Feature extractor** maintains indicator history and updates raw feature
  values as relevant events arrive. `FeatureExtractor` exposes these values
  through a feature vector, with IDs identifying the outputs.
* **Transformations** prepare raw features for a model. They can select or
  scale values, add lags, or apply supported preprocessing parameters fitted
  in Python, such as imputation values and PCA components.
* **Pipeline** combines the extractor and transformations. `PipelineSpec`
  describes the configuration; Rust's `Pipeline` and Python's
  `ModelInputPipeline` execute it and expose the final model-input vector.
  Your model integration consumes that vector to make predictions.

Feature-extractor and pipeline specifications can be serialized as JSON and
loaded in either language. A typical workflow is to define features and fit
supported transformations on historical data in Python, export the pipeline,
and rebuild it in Rust for live use. The JSON stores configuration and fitted
transformation parameters; indicator history and live order-book state must
be rebuilt by feeding events to the new runtime.

Supported features
------

### Indicators and market features

The following features are available through both the Rust and Python extractors.
Window sizes, sample lags, symbols, and applicable input sources are configurable.

| Category | Supported features |
| --- | --- |
| Returns | Simple and log returns over sample lags. |
| Moving averages | Simple moving average (SMA) over sample or time windows; exponential moving average (EMA) over sample periods. |
| Volatility | Rolling population standard deviation of simple returns over sample or time windows. |
| Trade activity | Rolling trade count, total trade volume, and volume-weighted average price (VWAP) over time windows. |
| Volume flow | Cumulative volume delta (CVD) over sample windows, on-balance volume (OBV) over time windows, and cumulative volume-price trend (VPT). |
| Calendar | UTC day of week and elapsed time since the first event of the day, with a configurable fixed UTC offset for day boundaries. |
| Order-book quotes | Best bid/ask prices and sizes, mid-price, spread, spread in basis points, weighted mid-price, and microprice. |
| Order-book depth | Size at a price, price or size at a chosen depth, cumulative depth through a price, price range and total size needed to reach a target size, volume between prices, and bid/ask imbalance. |

Timed indicators use configurable aggregation buckets. CVD requires the trade's
aggressor side; order-book features require a configured book fed with snapshots
and deltas.

### Transformations

Pipelines support scalar transformations and sequences of fitted preprocessing
stages. Supported Python estimators export their fitted parameters for execution
in Rust.

| Operation | Supported transformations |
| --- | --- |
| Scalar operations | Copy, rename, and reorder features; scale with supplied mean and scale; lag values by accepted-event count. |
| Fitted scaling | `StandardScaler`, `RobustScaler`, `MinMaxScaler`, and `MaxAbsScaler`. |
| Distribution transforms | `PowerTransformer` with Box-Cox or Yeo-Johnson; `QuantileTransformer` with uniform or normal output. |
| Missing values | `SimpleImputer` for NaNs using mean, median, most-frequent, or constant replacement, with optional missing-value indicators. |
| Feature selection | Explicit column selection and fitted `VarianceThreshold` selection. |
| Dimensionality reduction | PCA, with optional whitening. |

Python fitting uses the optional `fiml[sklearn]` extra. See the
[Python preprocessing documentation](crates/fiml-python/README.md#fitting-sklearn-stages)
for supported estimator options and restrictions.

Installation
--------

### Rust

Requires Rust **1.89 or newer** and a native linker for your platform.
From your application's directory, add Fiml directly from Git:

```bash
cargo add fiml --git https://github.com/pbezglasny/fiml --features serde
cargo add serde_json --features float_roundtrip
```

The `serde` feature enables JSON serialization of feature and pipeline
specifications. `serde_json` reads and writes those specifications;
`float_roundtrip` preserves fitted floating-point parameters when loading them.
For a Rust-only application that does not exchange JSON, omit `--features serde`
and the `serde_json` dependency.

### Python

Requires **CPython 3.12 or newer**. Installing from source also requires the
Rust toolchain and linker described above. Clone the repository, then create
a virtual environment:

```bash
git clone https://github.com/pbezglasny/fiml.git
cd fiml
python3 -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or macOS, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. From the repository root:

```bash
python -m pip install "./crates/fiml-python[pandas]"
python crates/fiml-python/examples/quickstart.py
```

The installation builds the Rust extension automatically and installs NumPy.
The optional `pandas` extra enables DataFrame input and output; omit `[pandas]`
when using only the NumPy API.

To also fit supported scikit-learn transformations:

```bash
python -m pip install "./crates/fiml-python[pandas,sklearn]"
```

The `sklearn` extra currently requires scikit-learn `>=1.9,<1.10`.
See the [Python installation guide](crates/fiml-python/README.md#install-from-source)
for development setup and troubleshooting.

Simple example
------

Compute a three-sample moving average, fit a scaler in Python, and export the
pipeline for Rust. This example requires the `sklearn` extra from Installation.

### Python: fit and export

Save this as `prepare_pipeline.py` and run `python prepare_pipeline.py`.
It writes `pipeline.json` in the current directory.

```python
from pathlib import Path

import fiml
import numpy as np
from sklearn.preprocessing import StandardScaler

raw = fiml.FeatureExtractorSpec().sma("BTCUSDT", [3], source="price")
spec = fiml.PipelineSpec(raw).identity(raw.feature_ids()[0], output="sma3")
pipeline = fiml.ModelInputPipeline(spec).add_transformation(
    StandardScaler(), name="scale"
)

prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
model_inputs = pipeline.fit_transform(
    kind=np.full(len(prices), fiml.KIND_PRICE, dtype=np.uint8),
    symbol=np.full(len(prices), pipeline.symbol("BTCUSDT"), dtype=np.int64),
    timestamp=np.arange(len(prices), dtype=np.int64) * 1_000,
    price=prices,
    fit_mask=np.arange(len(prices)) >= 2,  # Exclude the two warm-up rows.
)

assert np.isnan(model_inputs[:2]).all()
np.testing.assert_allclose(model_inputs[-1, 0], np.sqrt(1.5))
Path("pipeline.json").write_text(pipeline.to_json(), encoding="utf-8")
print(f"Final model input: {model_inputs[-1, 0]:.6f}")
```

The scaler is fitted on the ready SMA values `[102, 104, 106]`. The returned
matrix has one row per event, including the initial `NaN` rows. In a real
training workflow, fit preprocessing only on training data and use the frozen
pipeline for validation and live inference.

### Rust: load and replay

Save this as `src/main.rs` in the Cargo application configured in Installation.
Place `pipeline.json` in that application's root and run `cargo run` there.

```rust
use fiml::{ArrayFeatureVector, Event, PipelineSpec, Symbol};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string("pipeline.json")?;
    let spec: PipelineSpec = serde_json::from_str(&json)?;
    let mut pipeline = spec.build(
        ArrayFeatureVector::<1>::new(), // One raw SMA value.
        ArrayFeatureVector::<1>::new(), // One scaled model input.
    )?;
    let btc = Symbol::new("BTCUSDT")?;

    // Rebuild indicator history; fitted scaler parameters came from JSON.
    for (index, price) in [100.0, 102.0, 104.0, 106.0, 108.0].into_iter().enumerate() {
        pipeline.handle_event(Event::price(btc, price, index as i64 * 1_000))?;
        if index < 2 {
            assert!(pipeline.values()[0].is_nan());
        }
    }

    let model_input: &[f64] = pipeline.values();
    assert!((model_input[0] - 1.5_f64.sqrt()).abs() < 1e-12);
    println!("Final model input: {:.6}", model_input[0]);
    Ok(())
}
```

Both programs print `Final model input: 1.224745`. Continue feeding new price
events to the Rust pipeline and pass its borrowed output slice to your model.

Behavior and limits
-------

* **Warm-up and missing values:** With `WarmupPolicy::FullWindow`, windowed
  features produce `NaN` until enough samples or event time have accumulated.
  This is the Python builders' default; `FirstValue` allows partial-window
  results. Returns always require their exact sample lag. Simple returns need
  a nonzero lagged value; log returns need positive current and lagged values.
  Missing book levels and empty VWAP windows can also produce `NaN` after warm-up.
* **Event time:** Timestamps are Unix epoch milliseconds and must be
  nondecreasing per symbol across all event kinds. Different symbols may
  interleave; pipeline lags count accepted events in arrival order. Within an
  extractor, timed windows advance on events for their own symbol. Global
  `Time` events do not advance other symbols' windows. Standalone timed
  indicators expose `observe(timestamp)` to advance time without adding data.
* **Input validation:** Extractors and pipelines reject NaN and infinity in
  numeric event payloads and reject out-of-order timestamps. Standalone
  indicators rely on the caller to supply finite inputs and ordered time.
  Handle order-book update errors and supply a fresh snapshot when
  resynchronization is required.
* **State:** Processing a batch advances the same indicator history as
  processing its events individually. Start a fresh runtime before replaying
  an independent stream; Python's `ModelInputPipeline` also provides `reset()`.
  Loading JSON starts with empty event history while retaining fitted
  transformation parameters.
* **Output layout and precision:** Use feature IDs to identify columns rather
  than assuming builder-call order. Rust feature vectors use `f64`; Python
  returns `float64` by default and optionally casts outputs to `float32`.
  Reserved output cells remain `NaN` and are included in `values()`; active
  output IDs describe the cells intended for model input.
* **Allocation:** Built-in indicator and transformation updates reuse
  preallocated storage. Construction, JSON handling, Python output arrays and
  DataFrames, and order-book storage may allocate. Borrowing Rust output slices
  avoids copying their values.

The current configuration limits include:

| Limit | Scope |
| --- | --- |
| 512 symbol names | Per process, including the reserved global symbol. Names are ASCII-case-insensitive; slots are not reclaimed. |
| 16 compatible outputs | Per compiled indicator group, such as windows sharing the same indicator configuration. |
| 1–10,000 events | Each pipeline lag transformation's lookback. This limit does not apply to return-indicator sample lags. |
| Whole milliseconds | Timed aggregation intervals and windows; a window must be a positive multiple of its aggregation interval. |

### Version and platform support

The minimum versions are Rust 1.89 and CPython 3.12. Declared Python versions
include 3.12, 3.13, and 3.14. The release workflow targets Python wheels for
Linux x86-64/AArch64, macOS x86-64/AArch64, and Windows x86-64. Installation
from source requires a Rust toolchain and native linker. Release artifact
validation is tracked in the [release checklist](docs/release-check-list.md).

Other docs
----

* [Python guide](crates/fiml-python/README.md) — DataFrame and NumPy APIs,
  event replay, and fitted preprocessing.
* [Rust examples](crates/fiml/examples) and
  [Python examples](crates/fiml-python/examples) — runnable applications.
* [Feature-extractor JSON schema](docs/feature-extractor-spec.schema.json) and
  [pipeline JSON schema](docs/pipeline-spec.schema.json) — exported configuration
  formats.

Build the Rust API reference from the repository root:

```bash
cargo doc -p fiml --all-features --no-deps --open
```

License
-------

Fiml is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
for the full license text.

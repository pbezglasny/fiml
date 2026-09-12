# Fitted sklearn transformers in the FIML pipeline

Status: implemented.

Date: 2026-09-09. Updated: 2026-09-12 (QuantileTransformer).

Related: [issue #93](https://github.com/pbezglasny/fiml/issues/93),
[issue #118](https://github.com/pbezglasny/fiml/issues/118),
[issue #119](https://github.com/pbezglasny/fiml/issues/119),
[issue #120](https://github.com/pbezglasny/fiml/issues/120),
[issue #121](https://github.com/pbezglasny/fiml/issues/121),
[issue #122](https://github.com/pbezglasny/fiml/issues/122),
[issue #123](https://github.com/pbezglasny/fiml/issues/123), and
[issue #124](https://github.com/pbezglasny/fiml/issues/124).

Implementation: `fiml[sklearn]` supports scikit-learn `>=1.9,<1.10`; inference
requires no sklearn installation. See the [runnable example](../crates/fiml-python/examples/sklearn_pipeline.py)
and [Python API documentation](../crates/fiml-python/README.md#fitting-sklearn-stages).
The design below records the implemented contract and deliberately deferred scope.

## Recommendation

Fit supported sklearn transformers in Python, export their learned numeric state
into `PipelineSpec`, and execute that spec through the same Rust runtime in both
Python batch processing and online serving. Support `SimpleImputer`,
`StandardScaler`, `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`,
`PowerTransformer`, `QuantileTransformer`, `VarianceThreshold`, and `PCA`, including PCA whitening.
Add an ordered list of vector stages after the existing scalar transformations.

“Save parameters” must mean the fitted state needed for inference. Saving only
constructor arguments such as `n_components=2` cannot reproduce a trained PCA.
Its learned axes and mean are essential; its output width can also be determined
during fitting. See the [PCA attributes](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

This is support for exporting specific sklearn transformer types. An arbitrary
object with `fit` and `transform` cannot run in Rust merely because its parameters
are JSON serializable: Rust must implement its inference operation. Reject
unsupported types when added, before replaying training data.

## Starting point before issue #93

The relevant implementation is:

| Component | Current responsibility | Required change |
| --- | --- | --- |
| [Rust PipelineSpec](../crates/fiml/src/features/pipeline/specs.rs) | Validates scalar definitions and raw/final vector layouts | Add fitted vector stages and derive final width from the last stage |
| [Transformers](../crates/fiml/src/features/transformers/mod.rs) | Identity, fitted standard scaling, and shared lag histories; every input is raw | Preserve these operations as the base feature selection; add sequential vector execution |
| [Rust Pipeline](../crates/fiml/src/features/pipeline/mod.rs) | Extracts raw values and writes transformed values after accepted events | Execute fitted stages using storage allocated at construction |
| [Canonical JSON adapter](../crates/fiml/src/features/serde/pipeline_spec.rs) | Strict version `1.0`; transformation count equals final length | Introduce a versioned stage representation |
| [Python pipeline](../crates/fiml-python/python/fiml/__init__.py) and [binding](../crates/fiml-python/src/model_input_pipeline.rs) | Wrap an already compiled spec; expose event replay and DataFrame replay | Own an unfitted recipe, fit it, retain the fitted spec, and rebuild the runtime |

`ModelInputPipeline.transform(...)` already exists. It accepts event columns and
returns one model-input row per event, advancing indicator and lag state. It is
not sklearn's stateless `transform(X)` on a precomputed feature matrix. Add
`fit` and `fit_transform` with the same event-column convention and preserve
that existing `transform` contract.

[ADR 0008](adr/0008-serialize-the-pipeline-spec.md) already establishes the right
serialization boundary: persist the spec, not a live runtime. The current
[pipeline design](pipeline.md) defers chaining; this proposal introduces the
specific linear sequence needed for sklearn preprocessing.

## Python interface

Keep `PipelineSpec` as validated, deployable data. Unfitted sklearn objects belong
to the Python pipeline, because their learned state and sometimes their dimensions
do not exist yet. Extend the existing fluent construction style with
`add_transformation(estimator, *, name) -> self`; a second builder class or
`add_pca` convenience method is unnecessary initially.

Usage:

```python
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

import fiml

raw_spec = fiml.FeatureExtractorSpec().sma(
    "BTCUSDT", [2, 5, 10],
    source="trade_price",
    warmup=fiml.WarmupPolicy.FIRST_VALUE,
)
base_spec = fiml.PipelineSpec(raw_spec)
for feature_id in raw_spec.feature_ids():
    base_spec.identity(feature_id)

pipeline = (
    fiml.ModelInputPipeline(base_spec)
    .add_transformation(RobustScaler(unit_variance=True), name="scale")
    .add_transformation(PCA(n_components=2, whiten=True), name="pca")
)
events = dict(
    kind=np.full(12, fiml.KIND_TRADE, dtype=np.uint8),
    symbol=np.full(12, pipeline.symbol("BTCUSDT"), dtype=np.int64),
    timestamp=np.arange(12, dtype=np.int64),
    price=np.array([10., 12., 11., 15., 13., 18., 16., 20., 17., 23., 19., 25.]),
    volume=np.ones(12),
)

pipeline.fit(**events)                 # learns parameters; leaves runtime cold
X_train = pipeline.transform(**events) # Rust execution; advances runtime
# model.fit(X_train, y_train)
Path("pipeline.json").write_text(pipeline.to_json(), encoding="utf-8")

restored = fiml.ModelInputPipeline.from_json(
    Path("pipeline.json").read_text(encoding="utf-8")
)
# restored starts cold; register its symbols before supplying event arrays.
```

The base spec selects and orders stage inputs. It can contain identity, lagged,
or manually fitted scalar transformations. Reserved cells never enter sklearn.
Each added stage consumes the entire preceding active vector and replaces it.
The first stage consumes the base outputs; later stages consume previous stage
outputs. This supports scaler-to-PCA and `lags -> PCA` without branches,
arbitrary references, or a general transformation graph.

`fiml.ScalarStage()` groups can also appear anywhere in the sequence, authored
with `identity`, `lagged`, and fixed-parameter `standard_scale`. Each definition
reads the preceding layout; dependent definitions go into successive stages.
A scalar stage replaces its input with its authored outputs, so use identity
outputs for columns that must survive. Input resolution is deferred until the
stage is reached during fitting, after PCA output dimensions are known. Rust
represents these groups as `FittedStage::Scalar { transformations }`.

Scaler stages preserve input IDs. PCA outputs receive deterministic IDs
`<name>__pc0`, `<name>__pc1`, etc., after fitting resolves their count. Require
unique stage names and unique, nonreserved IDs within every layout. Raw and
successive layouts remain separate namespaces, so a scaler can preserve names.
Final `feature_names()`, `n_features()`, and `active_feature_count()` describe the
last stage, with the existing distinction between capacity and active length.

### Method and state contract

| Method | Behavior |
| --- | --- |
| `add_transformation(estimator, *, name)` | Check supported concrete type and options, snapshot an unfitted clone, append it, and invalidate the fitted result. Allow configuration before the first successful fit or event replay; create a new pipeline to change an established recipe. |
| `fit(kind, symbol, timestamp, *, price=None, volume=None, side=None, bid=None, ask=None, fit_mask=None)` | Replay from a fresh base runtime, fit cloned estimators sequentially, compile a candidate spec, and return `self`. On success install it with cold event state. |
| `transform(...)` | Keep the current event-column signature. Apply frozen parameters through Rust and return all event rows. Never learn or reset implicitly. |
| `fit_transform(..., fit_mask=None)` | Call `fit`, then replay the same events through `transform`. Return all rows, including warm-up rows. Fitting replays each completed prefix; the final replay returns all rows. |
| `reset()` | Rebuild from the fitted spec, clearing indicators, lag histories, books, and timestamps while retaining fitted parameters and existing symbol-handle mappings. |
| `to_spec()` / `to_json()` | Return a fitted spec snapshot / serialize it through the canonical Rust adapter. Reject an unfitted recipe. |
| `from_json(...)` | Load an inference-ready pipeline without importing sklearn. No training recipe is reconstructed, so `fit` on this object raises a clear inference-only error. |

A pipeline built directly from a fitted spec remains usable immediately. A
pipeline with pending estimators must reject inference, final-layout inspection,
and export until fitting succeeds; raw names and symbol registration remain
available. Repeated `fit` on a Python-authored recipe always starts from its
original unfitted templates, not from the previous fitted estimators.

Fit on temporary state and publish only after all training, numeric validation,
and Rust compilation succeed. A failed refit leaves the old fitted artifact,
symbol mapping, and live runtime intact. Clone estimator templates so fitting
does not mutate the user's objects; sklearn provides
[`clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
for this purpose. Preserve runtime-local symbol handles when creating the
temporary replay driver or replacing the live one: currently those handles index
`RuntimeDriver.symbols`, not a globally interchangeable integer namespace.

Keep `compute_features(df)` and `transform_order_book(events)` as existing
inference entry points into the fitted Rust runtime. The initial `fit` overload
above covers columnar events, not order-book payloads or DataFrames. Those need
explicit adapters if training on them is required; passing a DataFrame to
`transform` must not silently change the meaning of the existing method.
This also does not yet promise sklearn `Pipeline`/`GridSearchCV` compatibility:
that requires a separate estimator contract, cloning support, and chronological
fold handling.

## Training rows, warm-up, and replay

Replay every event through a fresh base pipeline before selecting training rows.
Skipping events would change moving averages and accepted-event lag histories.
Use the existing Rust event validation, per-symbol timestamp ordering, and arrival
order; never sort or shuffle the replay to prepare sklearn input.

Build a dense `float64` matrix from active base outputs. `fit_mask`, if supplied,
must be a one-dimensional boolean array with one entry per event. It selects
which snapshots train every stage, without suppressing the corresponding events.
With no mask, all snapshots are training rows. Require a nonempty selection and
finite values in every selected column, except that SimpleImputer, PowerTransformer,
and QuantileTransformer inputs may contain NaNs but not infinities. Report the first invalid
event row and feature ID. The caller can exclude indicator warm-up rows explicitly.
Do not silently replace NaNs or independently drop different rows outside that stage.

For each sklearn stage: clone, fit on selected rows, and export its numeric state.
For each scalar stage: validate and append its definitions. Replay all original
events from cold state through the completed Rust prefix to obtain the next
matrix. Do not mask before replay: even excluded rows must advance downstream
lags. Check selected estimator inputs and final outputs are finite. A scalar
selection may drop unused warm-up columns before the next estimator. This simple
implementation repeats extraction per prefix, with training cost proportional to
stage count; online execution still processes each event once. `copy=False`
remains unsupported.

Keep fitting in `float64`, even if public output arrays are configured as
`float32`; cast only the final returned arrays. Train the downstream predictor
on the result of FIML's Rust-backed `transform`, using the same dtype and active
column selection intended for serving.

`transform` retains every row and its alignment with labels. Scalar scaling
preserves its existing NaN propagation. For a PCA stage, any nonfinite input
makes that stage's entire output NaN; apply this guard explicitly, including
inputs with zero component weights. This is FIML's warm-up behavior, not a claim
that sklearn PCA accepts NaNs. Exclude these rows from downstream model training
and avoid using them for predictions until the required features are ready.

Fitting must use only the chronological training partition. Validation/test
events may continue after replaying the training prefix through the fitted
pipeline. For a separate stream, use `reset()` and supply the required historical
prefix. A JSON load restores fitted parameters, but cannot restore warmed
indicators, order-book synchronization, or lag history.

## Fitted state and numerical operations

Use a small explicit exporter dispatch for the exact `SimpleImputer`, `StandardScaler`,
`RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`, `PowerTransformer`,
`QuantileTransformer`, `VarianceThreshold`, and `PCA`
classes. Reject subclasses with potentially overridden behavior, arbitrary
callbacks, sparse outputs, and unsupported versions/options with a useful error.
There is no need for a plugin registry or a Rust dependency on sklearn.

### SimpleImputer

Support dense numeric input, `missing_values=np.nan`, `copy=True`, the four
built-in strategies, `keep_empty_features`, and `add_indicator`. Export only
the retained input indices, their finite replacement values, fitted indicator
input indices, and explicit output IDs. Rust replaces NaNs only; infinities pass
through unchanged and indicator values describe missingness before replacement.
Filling indicator warm-up is an explicit modeling choice and does not establish
feature readiness; `fit_mask` remains available to exclude those rows.

### StandardScaler

Export effective `mean[d]` and `scale[d]`. When `with_mean=False`, write zeros;
when `with_std=False`, write ones. Otherwise copy the fitted arrays. Use sklearn's
`scale_` directly, including its handling of constant features; do not recompute
it from the training data or assume the presence of `mean_` means centering is
enabled. See [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

The vector stage applies `(x[i] - mean[i]) / scale[i]`, using the existing scalar
scaling arithmetic and validation where practical. Store two vectors, not a
dense diagonal matrix. Its cost is O(d) time and O(d) fitted storage.

### RobustScaler

Export effective `center_[d]` and `scale_[d]` through the same vector stage.
When `with_centering=False`, write zeros; when `with_scaling=False`, write ones.
Otherwise use sklearn's fitted arrays directly so `quantile_range`,
`unit_variance`, constant columns, and outliers retain sklearn's behavior.
Unit-variance scaling requires `0 < q_min < q_max < 100`; reject other ranges
when the stage is added because sklearn produces a non-finite or zero fitted
scale for those configurations. The range is irrelevant when scaling is disabled.

### MinMaxScaler

Export sklearn's fitted `scale_[d]` and `min_[d]` directly. Apply
`x[i] * scale[i] + min[i]` in that order, then clamp to `feature_range` only
when `clip=True`. The stage preserves input IDs and NaNs. Scales must be positive
and finite; offsets and optional strictly increasing clip bounds must be finite.
This preserves sklearn's learned handling of constant and near-constant columns.
Without clipping, observations outside the fitted data range can exceed the
configured feature range.

### MaxAbsScaler

Export sklearn's fitted `scale_[d]` directly with zero centers through the
standard scaling stage when `clip=False`. When `clip=True`, reuse the MinMax
stage with reciprocal scales, zero offsets, and `[-1, 1]` clipping. Both paths
preserve input IDs, zeros, signs, and NaNs in O(d) time and storage. Without
clipping, future magnitudes may exceed `1`; fitting is not robust to outliers.

### PowerTransformer

Export the method, fitted `lambdas_[d]`, and effective post-transform `mean[d]`
and `scale[d]`; disabled standardization uses zeros and ones. Derive scaling by
applying SciPy's public transform functions and fitting a public `StandardScaler`,
without reading sklearn's private scaler. Rust uses stable `log1p`/`expm1`
forms for Yeo-Johnson and Box-Cox in O(d) time and storage. NaNs are preserved.
Box-Cox rejects nonpositive fitting values; frozen inference writes NaN only for
an affected nonpositive column, rather than rejecting the event as sklearn does.

### QuantileTransformer

Export fitted `quantiles_[q][d]`, `references_[q]`, the output distribution,
sklearn's `1e-7` boundary threshold, and its finite normal clipping bounds. Validate
finite rectangular tables, matching dimensions, nondecreasing quantiles per column,
and strictly increasing reference probabilities starting at `0` and ending at `1`
when `q > 1`; repeated quantiles are
valid. Compile the row-major artifact once into contiguous per-column tables.

An all-NaN fitted column has no finite quantile table. Export its input index and
a finite placeholder column; Rust then reproduces sklearn's NaN propagation so a
following imputer can fill it. With the degenerate `q = 1`, finite inference values
map to `0` (or the lower normal clip), matching sklearn's single-reference behavior.

For each value, binary-search both sides of a repeated quantile and average the two
linear interpolations, matching sklearn's tie behavior in O(log q) work. Preserve
NaNs and saturate values outside the fitted range. Uniform output stops at the
interpolated probability. Normal output applies Wichura's AS 241 inverse-normal
rational approximation, accurate to floating-point precision across the supported
clipped range, then applies the exported finite tail bounds. This costs O(d*q)
fitted storage and no per-event allocation. The nonlinear rank mapping changes
linear correlations and should be chosen deliberately.

Representative normal-output measurements on an AMD Ryzen 9 9900X with the
committed Criterion benchmark (`cargo bench -p fiml --bench quantile_transform
-- --quick`) are:

| Features (`d`) | Quantiles (`q`) | Compact JSON artifact | Median per row |
| ---: | ---: | ---: | ---: |
| 8 | 64 | 21,718 bytes | 152 ns |
| 32 | 256 | 274,199 bytes | 667 ns |

The artifact measurement uses short output IDs and includes the complete pipeline
JSON. Latency includes one raw feature update, base-vector expansion, binary
lookups, and normal conversion. Both scale as the stored O(d*q) tables and
O(d*log(q)) lookup path predict; compare again on deployment hardware before
using these numbers for capacity planning.

### VarianceThreshold

Require a finite, nonnegative `threshold` and finite selected training inputs.
After fitting in Python, export `get_support(indices=True)` as a nonempty,
strictly increasing index vector. A generic selection stage stores only those
indices and the matching retained input IDs; it does not store variances or
compute statistics in Rust. Inference copies O(k) retained values into
preallocated output, preserving retained NaNs and ignoring discarded columns.
Positive thresholds depend on input scale. This is neither correlation-based nor
supervised feature selection.

### PCA

Export `mean[d]`, `components[k][d]`, and `output_scale[k]`. Matrix rows are output
components and columns follow the preceding stage's exact input order. `k` is
the fitted component count, not necessarily the original `n_components` value.

Define the wire operation as:

```text
projected_mean[j] = sum_i(mean[i] * components[j][i])  # computed at build
output[j] = (sum_i(input[i] * components[j][i]) - projected_mean[j])
            / output_scale[j]
```

For ordinary PCA, `output_scale` contains ones. For whitening, export
`max(sqrt(explained_variance_[j]), epsilon_float64)`. The inspected sklearn
implementation projects first, subtracts the projected mean, then divides by this
clipped scale. Use its public fitted attributes to export; do not call private
methods. See the [PCA transform implementation](https://github.com/scikit-learn/scikit-learn/blob/cc50648cc1b759b53a4edbce0f3bb6c237349448/sklearn/decomposition/_base.py).

Saving the effective divisor makes clipping part of the numeric artifact instead
of requiring Rust to infer a sklearn version's policy. Keep division after the
projection rather than folding it into the component matrix. Validate finite
means, components, projected means, and positive finite divisors. Near-zero
variance can amplify rounding differences even with clipping. The rank-deficient
regression covers both `full` and `covariance_eigh`: normal components use
`rtol=1e-10, atol=1e-12`, while nearly null components use a forward-error bound
based on input magnitudes, component weights, and the whitening divisor.
Whole-unit differences from sklearn are possible in those nearly null components;
there is no uniform small absolute-error guarantee for whitening arbitrary data.
Reduce the retained components or disable whitening if these differences matter.
Train the predictor on FIML's returned matrix: artifact reloads reproduce that
matrix exactly under the same Rust runtime, including these degenerate cases.

PCA inference needs no SVD, covariance fitting, or training samples. A nested
loop over contiguous row-major weights costs O(k*d) per event and O(k*d + d + k)
fitted storage. Start without BLAS, a matrix library, SIMD specialization, or
stage fusion. Benchmark representative dimensions before adding those.

## JSON format and validation

Introduce pipeline format `2.0`, keeping the nested feature extractor at `1.0`.
New readers should accept existing strict `1.0` artifacts as having no stages;
writers emit `2.0` for these fitted-only sequences. Scalar stages require `2.1`,
which readers also accept. Their representation is
`{"type": "scalar", "transformations": [...]}`, reusing the base transformation
wire format. A `2.0` artifact containing a scalar stage is rejected.
MinMaxScaler and clipped MaxAbsScaler stages use `2.2`; older versions reject
the reused stage tag.
SimpleImputer stages use `2.3`; older versions reject the stage tag.
PowerTransformer stages use `2.4`; older versions reject the stage tag.
VarianceThreshold selection stages use `2.5`; older versions reject the stage tag.
QuantileTransformer stages use `2.6`; older versions reject the stage tag.
Do not reinterpret `1.0`, whose reader currently requires final length to equal
the scalar transformation count.

`2.0` `model_input` fragment, assuming raw IDs `raw_a` and `raw_b`:

```json
{
  "capacity": 1,
  "length": 1,
  "transformations": [
    {"type": "identity", "input": "raw_a", "output": "a"},
    {"type": "identity", "input": "raw_b", "output": "b"}
  ],
  "stages": [
    {
      "type": "standard_scale",
      "outputs": ["a", "b"],
      "mean": [10.0, 20.0],
      "scale": [2.0, 4.0]
    },
    {
      "type": "pca",
      "outputs": ["pca__pc0"],
      "mean": [0.0, 0.0],
      "components": [[0.6, 0.8]],
      "output_scale": [1.0]
    }
  ]
}
```

Stage inputs are implicit: the complete preceding active layout, in order.
Base scalar definitions resolve stable raw IDs; scalar stages resolve IDs against
the preceding layout. Explicit fitted stage output IDs
freeze the layout across export/import. The example has two base outputs and
one final output; base scratch width must not be taken from final `capacity`.

Require `stages` in `2.0` through `2.6`, permitting an empty list. Derive base
width from scalar definitions, each stage width from its validated arrays and outputs, and final
active length from the last stage (or base width if empty). Final `capacity`
must cover that final length; it may be smaller than an intermediate width.
For Python authoring, default final capacity to the fitted final width; any
explicit capacity on the supplied base spec remains an explicit final capacity
constraint and must be revalidated after fitting.

Validate the whole artifact on construction: strict versions/fields/tags,
existing scalar rules, nonempty stage input/output layouts, dimension agreement,
rectangular matrices, unique nonreserved output IDs, and finite numeric state.
Scaler outputs must preserve preceding IDs; PCA must have `1 <= k <= d`.
Use checked dimension arithmetic before allocating scratch or flattening matrices.
Report the stage index and relevant ID/dimension. Unknown stages must fail rather
than be skipped. JSON numbers use full `f64` precision; NaN/Infinity are forbidden
in fitted artifacts.

For exact float restoration, Rust applications using serde_json must enable its
`float_roundtrip` feature. The Python binding and maintained Rust fixtures do
this; the default parser can round some fitted numbers by one ULP.

Persist only inference state and existing metadata. Do not serialize Python
objects, pickle/joblib data, training rows, estimator `__dict__`, solver choices,
random generators, or live event state. Keep training configuration and package
versions in the training experiment if refitting must be reproducible. Existing
checksums remain opaque metadata, not integrity validation.

## Rust and Python implementation boundaries

Use a concrete stage enum with scaling, PCA, and scalar-group variants beside the
base scalar definitions. Keep `PipelineSpec` as the semantic validator and canonical
serde owner. Add a separate compiled-stage enum using flattened numeric arrays;
there is no need for dynamic dispatch or a public transformer trait yet.

For pipelines with stages, allocate a base vector and intermediate scratch at
`build`. Reuse two work buffers sized for the required intermediate widths,
alternating source and destination; the last stage writes into the caller-owned
model vector. Ensure PCA never overwrites inputs that another component still
needs. Keep the existing direct-write path for pipelines with no stages.
Reserved final cells stay NaN, and processing a rejected event does not execute
stages or advance lag histories; retain existing order-book error semantics.

The steady-state event path uses numeric indexes and preallocated buffers, with
no heap allocation, JSON work, Python calls, or string lookups. Configuration,
fitting, and Python batch-output allocation are outside that guarantee.

The Python layer should own the base spec, estimator templates, fitted spec, and
a replaceable Rust runtime. Prefer composition for the public Python wrapper:
the current PyO3 subclass eagerly embeds a runtime whose final width cannot be
known before PCA fitting. Forward the existing runtime methods, dtype validation,
and DataFrame helper; keep event validation/replay in the existing bindings.
Retain symbol registration order across temporary runtimes, reset, and refit.
Guard every inference entry point while unfitted, including `update`, `values`,
DataFrame replay, and order-book replay.

Add sklearn as an optional Python extra, for example `fiml[sklearn]`, and import
it only for authoring/fitting. Extend the existing test dependencies/CI setup to
install a declared, tested sklearn version range. Loading and serving an artifact
must work with the current base NumPy dependency and no sklearn installation.
Choose that range from passing parity tests, not an assumption that all sklearn
releases have identical numerical behavior.

## Implementation order and acceptance checks

1. Add stage definitions, validation, compilation, final-layout derivation, and
   Rust inference. Preserve scalar-only behavior and the existing allocation test.
2. Add the `2.0` adapter and schema, explicit `1.0` reader support, and Python spec
   bindings for fitted arrays. Update output-ID collection, which currently assumes
   one final output per scalar definition.
3. Add Python recipe ownership, exporter dispatch, `fit`, `fit_transform`, reset,
   and export. Exercise one scaler-to-PCA workflow before expanding adapters.
4. Add a maintained example and update the pipeline documentation/ADR to describe
   stages, training state, and format migration.

Use the existing tests and parity-fixture structure. Completion requires:

- Fit real sklearn estimators in Python, export JSON, load it independently in
  Rust, and compare event-by-event outputs against sklearn `fit` then `transform`
  on the same base rows. Also compare the Python Rust-backed replay and reloaded
  pipeline; a Rust-to-Rust comparison alone would miss exporter mistakes.
- Cover scaling flags and constant columns; PCA dimension reduction, fitted
  component count, whitening, rank-deficient data, and large feature offsets;
  chained scaling/PCA; and PCA consuming lagged base outputs.
- Specify tolerances for `float64` and final `float32` casts. Start ordinary
  well-conditioned fixtures at `rtol=1e-10, atol=1e-12` for `float64`, then assess
  ill-conditioned cases separately. BLAS reduction order prevents a general
  bitwise-equality promise; do not silently loosen all tests to hide one failure.
- Verify NaN warm-up and explicit fit masks, retained row alignment, cold state
  after fit/load, chronological chunk continuation, reset, stable symbol handles,
  failed-refit atomicity, unfitted errors, and unsupported estimator rejection.
- Verify dimensions/IDs and malformed numeric JSON, `1.0` loading and `2.0`
  round-trips, changed final capacity, reserved cells, and inference without sklearn.
- Extend [pipeline_allocations.rs](../crates/fiml/tests/pipeline_allocations.rs) to
  assert zero allocations through scaling/PCA warm-up and steady-state execution.

Verification uses `cargo fmt --all`, the affected Rust/Python tests,
`cargo clippy --workspace --all-targets --all-features -- -D warnings`, and the
existing project checks. The runnable coverage lives in
[test_sklearn_pipeline.py](../crates/fiml-python/tests/test_sklearn_pipeline.py),
[pipeline_stages.rs](../crates/fiml/tests/pipeline_stages.rs), and the allocation
test linked above. The [shared fitted fixture](../tests/fixtures/sklearn_pipeline.json)
is generated from the maintained sklearn example and independently replayed by Rust.

Defer other transformer exporters, `ColumnTransformer`, arbitrary sklearn
pipelines, supervised transformers, `partial_fit`, Rust training, and runtime
checkpoints until a concrete use case requires them. Each future exporter needs
a defined inference operation and a Python/Rust parity test before it is accepted.

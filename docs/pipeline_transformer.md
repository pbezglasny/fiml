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
Attach vector and scalar stages to their explicit parent nodes.

“Save parameters” must mean the fitted state needed for inference. Saving only
constructor arguments such as `n_components=2` cannot reproduce a trained PCA.
Its learned axes and mean are essential; its output width can also be determined
during fitting. See the [PCA attributes](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

This is support for exporting specific sklearn transformer types. An arbitrary
object with `fit` and `transform` cannot run in Rust merely because its parameters
are JSON serializable: Rust must implement its inference operation. Reject
unsupported types when added, before replaying training data.

## Python graph interface

Author scalar and estimator nodes with `PipelineSpec.add(parent, stage)`, then
register named outputs and call `build()`. Reusing a parent handle shares its
fitting and runtime state. Each insertion snapshots the supplied configuration.

```python
import fiml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

raw = (fiml.FeatureExtractorSpec()
       .field("BTCUSDT", source="trade_price", id="price")
       .field("BTCUSDT", source="trade_volume", id="volume"))
spec = fiml.PipelineSpec(raw)
scaled = spec.add(spec.root(), StandardScaler())
projected = spec.add(scaled, PCA(n_components=2))
spec.output("model", projected)
spec.output("decision", spec.root())
pipeline = spec.build()
# Register symbols, then supply the usual event columns.
# pipeline.fit(**training_events, fit_mask=training_rows)
# matrices = pipeline.transform(**following_events)
# latest = pipeline.output("model").values()
```

The [graph API](multiple_output.md) defines ownership, output handles, dictionaries
of batch results, transactional fit, reset, and serialization. Topology is fixed
in built pipelines. There is no `add_transformation` method on the runtime.
Artifacts are inference-only and do not reconstruct estimator recipes. Training
accepts columnar events and optional context schedules; order-book and DataFrame
training adapters are not included. Inference supports both replay paths.

## Training rows, warm-up, and replay

Replay every event through a fresh base pipeline before selecting training rows.
Skipping events would change moving averages and accepted-event lag histories.
Use the existing Rust event validation, per-symbol timestamp ordering, and arrival
order; never sort or shuffle the replay to prepare sklearn input.

Build a temporary dense `float64` matrix from the estimator parent's active outputs. `fit_mask`, if supplied,
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
matrix. Shared estimators fit once per operation; no graph-wide training cache is
retained. Do not mask before replay: even excluded rows must advance downstream
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

Pipeline format `4.0` stores an ordered node array and a separate ordered output
array; nested extractor format remains `2.0`. Stage payloads contain the numeric
fields described above. PCA component IDs are local `pc0`, `pc1`, etc. Other
operations retain their established naming rules within the node layout.

See the [complete graph example](multiple_output-arc.md#artifact-format-and-compatibility)
and [JSON Schema](pipeline-spec.schema.json). Rust validates parent indexes,
feature references, dimensions, positive output widths, and finite numeric
parameters before exposing a runtime. Unknown fields and unsupported versions
are rejected. No 3.0 compatibility reader is provided.

Numeric inference is implemented by the existing Rust stages; Python only fits
supported estimators and exports their state. Loading, graph evaluation, and
numeric inference require no sklearn, BLAS, or matrix-library dependency in Rust.
Enable serde_json's `float_roundtrip` feature when consuming fitted artifacts.
Python/Rust fixture parity, independent sklearn oracles, transactional refits,
branching histories, cold reload, and allocation tests cover the graph contract.

# Multiple pipeline outputs

A pipeline owns one extractor and an append-only graph of transformation stages.
Every stage consumes one parent vector and may feed several children. Reusing a
node handle shares computation, fitted parameters, and history. Registering an
output exposes an existing node; it does not create another calculation.

The recorded decisions and rationale are in
[multiple_output-arc.md](multiple_output-arc.md). Multiple-parent merges,
automatic deduplication, runtime topology mutation, per-output fitting masks,
and scratch-buffer reuse are deferred.

## Python authoring and results

```python
import fiml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

raw = (fiml.FeatureExtractorSpec()
       .field("BTCUSDT", source="trade_price", id="price")
       .field("BTCUSDT", source="trade_volume", id="volume"))
spec = fiml.PipelineSpec(raw)
root = spec.root()
projected = spec.add(root, PCA(n_components=2))
scaled = spec.add(projected, StandardScaler())
selected = spec.add(projected, fiml.ScalarStage().identity("pc0"))
spec.output("model_a", scaled)
spec.output("model_b", selected)
spec.output("decision", root)
pipeline = spec.build()
```

`add(parent, stage)` accepts a `ScalarStage`, a supported sklearn estimator, or a
numeric dictionary in the fitted-stage JSON format. For example, a frozen PCA
stage contains `type`, `outputs`, `mean`, `components`, and `output_scale`.
Insertion copies scalar/numeric configurations and clones estimators as unfitted
recipes. Later caller mutations do not change the builder. Each `build()` creates
an independent snapshot. Node handles belong to their originating builder.

A pipeline with estimator recipes requires `fit()` before inference or export.
Register symbols using `pipeline.symbol(name)` before preparing event arrays.

```python
# training_events and following_events are the usual named event arrays.
pipeline.fit(**training_events, fit_mask=training_rows)
model_a = pipeline.output("model_a")
batch = pipeline.transform(**following_events)
X_a = batch["model_a"]
X_b = batch["model_b"]
latest_a = model_a.values()
```

`transform`, `fit_transform`, and `transform_order_book` return dictionaries in
output-registration order, including when only one output exists. Each matrix
has one row per event and only active columns. `compute_features` returns a
corresponding dictionary of DataFrames. Empty event batches produce zero-row
matrices with the configured widths. Online `update` refreshes all nodes without
creating batch matrices.

`PipelineOutput.values()` returns an independent NumPy snapshot. Its
`feature_names()`, `n_features()`, and `active_feature_count()` describe the
current fitted layout. The handle remains valid across reset and refitting;
previously returned arrays remain independent. There is no pipeline-level
`values()` or default output. `output_names()` enumerates registered names.

Output names must be unique, nonblank, and outside `__reserved_`. A runnable
pipeline requires at least one output, and each must contain at least one active
feature. Several names may refer to the same node. Feature IDs are local to each
node; generated PCA names are `pc0`, `pc1`, etc. Downstream stages resolve IDs only
against their direct parent. Missing learned coordinates fail fit validation.

## Rust construction and borrowing

```rust
use fiml::{FeatureExtractorSpec, FeatureId, FittedStage, PipelineSpec, TransformerDefinition};

fn build(raw: FeatureExtractorSpec) -> fiml::Result<fiml::Pipeline> {
    let mut spec = PipelineSpec::new(raw);
    let root = spec.root();
    let lag = spec.add(root, FittedStage::Scalar {
        transformations: vec![TransformerDefinition::lagged(
            FeatureId::new("price"), FeatureId::new("previous_price"), 1,
        )],
    })?;
    spec.output("model", lag)?;
    spec.output("decision", root)?;
    spec.build()
}
```

`pipeline.output(name)` returns an optional borrowed `PipelineOutput`. Its
`values()`, `feature_ids()`, and `observations()` borrow the current node buffer.
Retain `output.id()` across event updates and resolve it with
`pipeline.output_by_id(id)`. IDs from another pipeline are rejected. Value borrows
must end before mutation. `reset()` clears histories while retaining parameters
and output IDs. `raw_values()` includes extractor reserved cells for diagnostics;
root outputs exclude those cells.

The runtime owns one preallocated result buffer per transformation node and the
extractor's existing storage. Consumers read parent buffers without modifying
them. Shared nodes run once per accepted event. Graph evaluation and borrowed
Rust reads allocate nothing; existing order-book storage is a separate concern.

## Training and lifecycle

One shared `fit_mask` selects training rows for all estimators. Each shared
estimator is fitted once per fit operation. Separate nodes fit independently,
even when authored from the same estimator object. Excluded rows still replay to
preserve lag and rolling histories. Fitting uses repeated cold event replays as
needed, without a graph-wide cache of training matrices. A shared node executes
once per event within each replay, not necessarily once over the entire fit.

The runnable [training example](../crates/fiml-python/examples/multiple_outputs.py)
fits a shared median imputer over SMA/lag snapshots, then independent
StandardScaler/PCA and MinMaxScaler/PCA branches, alongside a scalar decision
output. It applies one mask, checks every row against independent NumPy/sklearn
references, and exports a numeric artifact for replay without sklearn. The shared
[fitted fixture](../tests/fixtures/sklearn_multiple_outputs.json) verifies those
outputs in both Python and standalone Rust with `rtol=1e-10`, `atol=1e-12`.

A failed fit preserves the previous fitted artifact and live runtime. A successful
fit installs cold state. Loading an artifact also starts cold. `transform`
advances the live runtime; call `reset()` before replaying an independent stream.
Context updates, source clocks, rejected events, and book resynchronization retain
the existing event-processing semantics.

## Artifacts and migration

Pipeline JSON version `4.0` requires ordered `nodes` and `outputs` arrays. Node zero
is the extractor; each transformation references an earlier parent index. Nested
extractor specifications retain version `2.0`. Numeric stage payloads are unchanged.
The [schema](pipeline-spec.schema.json) checks record shapes; Rust also validates
references, dimensions, output layouts, and numeric parameters. Node creation and
output registration order survive round trips.

Export configuration and fitted numeric parameters only. Builder/runtime ownership
IDs, live values, clocks, histories, training rows, and estimator objects are not
serialized. Version `3.0` artifacts must be recreated and exported using the new
builder; no compatibility reader or automatic migration is provided.

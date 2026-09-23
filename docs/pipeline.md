# Pipeline status

Status: in progress

Last updated: 2026-09-18

The pipeline runtime, model-input serialization, and Python interface are
implemented, including Python-fitted sklearn `SimpleImputer`, `StandardScaler`,
`RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`, `PowerTransformer`,
`VarianceThreshold`, and PCA
stages with Rust online inference.
Remaining interface-hardening items are listed below.

## Current interface

The Rust runtime intentionally exposes a small interface:

```rust,ignore
pipeline.handle_event(event)?;
pipeline.raw_values();
pipeline.values();
pipeline.output_ids();
pipeline.last_timestamp();
pipeline.last_timestamp_for_symbol(symbol);
```

`PipelineSpec` compiles one `FeatureExtractorSpec` and an authored sequence of
scalar transformations into a base vector. Each scalar reads the raw feature
vector. Optional fitted vector stages then consume the preceding active layout
in sequence. `FittedStage::StandardScale` preserves its width and IDs;
`MaxAbsScaler` reuses it without clipping and the MinMax stage with clipping.
`FittedStage::SimpleImpute` stores retained inputs, finite replacements, and
fitted missing-indicator inputs.
`FittedStage::PowerTransform` applies Box-Cox or Yeo-Johnson and effective scaling.
`FittedStage::Select` copies retained columns by prevalidated numeric index.
`FittedStage::Pca` projects to named components and supports whitening.
`FittedStage::Scalar` selects, renames, scales, averages, or lags columns from the preceding
layout. Definitions within a scalar stage are independent; dependent operations
belong in successive stages. Intermediate widths may exceed final capacity.
General transformation graphs are not supported.

Python `ModelInputPipeline.add_transformation`, `fit`, and `fit_transform`
train supported sklearn objects and freeze numeric parameters.
`fiml.ScalarStage()` builders can be inserted anywhere in that sequence. Training
replays every event through each completed prefix, including rows excluded by
`fit_mask`, so downstream lag history matches inference. `transform`
continues to replay events through Rust. `reset()` clears event state while
retaining parameters and symbol handles. See [the transformer design](pipeline_transformer.md)
and [the runnable example](../crates/fiml-python/examples/sklearn_pipeline.py).

Lagged definitions for the same input within each scalar stage (or the base
layout) compile into one transformer with one
history buffer sized to the largest lag. Each output keeps its authored position
and becomes available after its own positive lag window. Duplicate lag windows
with different output IDs are supported; rejected events do not advance history.

Timestamps are nondecreasing per symbol across all event kinds. Symbols may
interleave with older timestamps; transformations and lag histories still follow
accepted-event arrival order. `last_timestamp()` reports the last accepted arrival,
while `last_timestamp_for_symbol(Symbol)` reports that symbol's ordering watermark.
Timed windows advance only on their symbol's events. Global calendar features
follow the maximum accepted timestamp, and global Time ticks do not advance other
symbols' windows. Python `update`, `transform`, and `compute_features` share these
rules; batches validate timestamp ordering before replaying any rows.

The canonical artifact has three ownership levels:

```json
{
  "version": "3.0",
  "checksum": "optional model metadata",
  "feature_extractor": {
    "version": "2.0",
    "capacity": 2,
    "length": 2,
    "required_events": [],
    "features": []
  },
  "model_input": {
    "capacity": 2,
    "length": 2,
    "transformations": [],
    "stages": []
  }
}
```

`feature_extractor` owns the raw-vector layout. `model_input` owns the final
vector layout, scalar base transformations, and fitted vector stages. Writers
emit pipeline `3.0` with extractor `2.0`; readers accept only these versions.
[Sample-average migration](sample-average-migration.md) describes replacing extractor
SMA/EMA with field extraction and scalar transformations.
The envelope above illustrates ownership; populated feature/transform arrays
must agree with the declared lengths. `required_events` is the canonical,
deduplicated list of concrete symbol/event inputs consumed by the extractor.
The strict source spelling for a
feature that observes any event is `any_event`.

## Open issues

### P3: Transformation validation diagnostics

Status: partial

Priority: medium

Validation reports the transformation index and a typed reason. It does not
include the offending input and output feature IDs, which makes large fitted
artifacts harder to diagnose.

Completion criteria:

- Include the transformation index, input ID, and output ID in validation
  errors.
- Preserve typed, allocation-free runtime errors; construction-time allocation
  is acceptable.
- Verify equivalent error context through Rust and Python JSON loading.

### P4: End-to-end failure and warm-up examples

Status: partial

Priority: medium

The Python example demonstrates successful JSON restoration and standard
scaling. User-facing examples do not yet demonstrate warm-up `NaN` behavior or
validation/rejected-event failures.

Completion criteria:

- Add or extend a Rust model-input replay example.
- Demonstrate warm-up `NaN`, successful scaling, and rejected-event atomicity.
- Keep the example driven by the same canonical artifact used for parity
  verification.

### P5: Python spec-builder cloning

Status: open

Priority: low

Each Python transformation append clones the existing definitions, nested raw
spec, and checksum, then revalidates the complete candidate. Repeated appends
therefore perform quadratic cold-path work.

Completion criteria:

- Accumulate Python transformations without cloning the complete spec for each
  append.
- Perform full semantic validation when producing or compiling the final spec.
- Preserve atomic failure behavior and the fluent Python interface.

### P6: Internal Python runtime documentation

Status: open

Priority: low

The internal `RuntimeDriver` and `RuntimeLayout` structs are undocumented,
despite owning the shared feature-extractor/model-input replay behavior and
layout metadata.

Completion criteria:

- Document why each struct exists and what invariant it owns.
- Avoid expanding the public runtime interface.

## Verification status

Verification for fitted stages:

- `make test` (Rust tests, Python tests, and the maintained notebook);
- real sklearn training/export and independent Rust fixture replay;
- exact fitted JSON reloads and zero-allocation stage execution;
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`.

## Deferred work

Do not add general transformation graphs, runtime state serialization, or
speculative transformer exporters. Add another exporter only for a concrete
training requirement, with a defined Rust inference operation and parity test.

## Caller-supplied context

`FeatureKey::Context { symbol, name }` registers a named precomputed scalar using
ordinary `FeatureDefinition` IDs and output slots. `Symbol::GLOBAL` provides
shared scope. Context has no event subscriptions or indicator calculator.

```rust,ignore
let key = FeatureKey::Context {
    symbol: Symbol::GLOBAL,
    name: "previous_day_high".to_owned(),
};
let definition = FeatureDefinition::new(key, FeatureId::new("previous_day_high"));
// Include definition in the extractor spec before constructing the pipeline.
pipeline.update_context(&[("previous_day_high", Some(105.0))])?;
pipeline.update_context(&[("previous_day_high", None)])?; // clear to NaN
```

Both `FeatureExtractor::update_context` and `Pipeline::update_context` validate
all IDs and values before writing. IDs must identify context outputs and may not
repeat within a call. Numeric values must be finite. Omitted values are retained;
an empty update is a no-op. Updates and pipeline refreshes allocate nothing after
construction. `validate_context` exposes the same validation without mutation for
batch replay preparation.

A context refresh executes stateless stages through the entire pipeline while
stateful transformers emit retained outputs. It does not alter event timestamps,
calendar features, order books, indicator histories, or lag history. Context
slots never set observation flags, so SMA/EMA do not average context replacements.
Event lags can sample held context values on subsequent accepted events.

The additive JSON kind retains extractor version `2.0` and pipeline version `3.0`:

```json
{
  "kind": "context",
  "source": {"type": "context"},
  "options": {"name": "previous_day_high"},
  "outputs": [{"id": "previous_day_high"}]
}
```

Place this indicator under its symbol group. Omit `outputs` to use the default ID,
which encodes symbol and context-name lengths and contents. Names must be nonempty.
Context contributes no `required_events` entries. Artifacts store definitions,
not held values; constructing a runtime initializes context to NaN. See the
[Python context examples](../crates/fiml-python/README.md#externally-supplied-context)
for scheduled training replay and chunked DataFrames. Callers supply precomputed
values when available and own stale-value clearing and session boundaries.

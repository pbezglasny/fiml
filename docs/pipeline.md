# Pipeline status

Status: in progress

Last updated: 2026-09-10

The pipeline runtime, model-input serialization, and Python interface are
implemented, including Python-fitted sklearn `StandardScaler`, `RobustScaler`,
`MinMaxScaler`, and PCA stages with Rust online inference. Remaining interface-hardening items
are listed below.

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
`FittedStage::Pca` projects to named components and supports whitening.
`FittedStage::Scalar` selects, renames, scales, or lags columns from the preceding
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
  "version": "2.0",
  "checksum": "optional model metadata",
  "feature_extractor": {
    "version": "1.0",
    "capacity": 2,
    "length": 2,
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
emit `2.2` for specs containing MinMaxScaler stages, `2.1` for scalar stages,
and `2.0` otherwise; readers also accept strict `1.0` artifacts without stages.
The envelope above illustrates ownership; populated feature/transform arrays
must agree with the declared lengths. The strict source spelling for a
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

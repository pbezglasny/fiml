# Pipeline status

Status: in progress

Last updated: 2026-09-04

The pipeline runtime, model-input serialization, and Python interface are
implemented. The remaining work is verification and interface hardening rather
than additional transformation types.

## Current interface

The Rust runtime intentionally exposes a small interface:

```rust,ignore
pipeline.handle_event(event)?;
pipeline.raw_values();
pipeline.values();
pipeline.output_ids();
```

`ModelInputSpec` compiles one `FeatureVectorSpec` and an authored sequence of
scalar transformations into a `Pipeline`. Every transformation reads directly
from the raw feature vector. Transformation chaining and general graphs are not
supported.

The canonical artifact has three ownership levels:

```json
{
  "version": "1.0",
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
    "transformations": []
  }
}
```

`feature_extractor` owns the raw-vector layout. `model_input` owns the final
vector layout and ordered transformations. The strict source spelling for a
feature that observes any event is `any_event`.

## Completed

- `Identity` and per-feature `StandardScale` transformation definitions.
- Cold-path validation of raw input IDs, duplicate and reserved output IDs,
  vector dimensions, and fitted scaler parameters.
- Compilation of stable IDs into numeric indexes and contiguous scalar
  operations.
- Direct writes into caller-owned model-vector storage after each accepted
  event.
- `NaN` initialization and propagation through identity and standard scaling.
- Rejected events leave raw and final snapshots unchanged.
- Stable authored output ordering and separate raw/final ID layouts.
- Strict, versioned `ModelInputSpec` JSON serialization behind the `serde`
  feature.
- One canonical Rust serialization adapter shared by the Python bindings.
- Python construction, JSON loading, event updates, array replay, and DataFrame
  replay for model-input pipelines.
- Replacement of the stale legacy pipeline.

## Open issues

### P1: Shared Rust/Python runtime parity fixture

Status: complete

Priority: high

The manually authored fixture lives in
[`tests/fixtures/model_input_parity`](../tests/fixtures/model_input_parity). Its
strict `1.0` model-input artifact, two `BTCUSDT` trades, time event, and literal
expected snapshots are consumed independently by the Rust contract test in
[`crates/fiml/tests/model_input_parity.rs`](../crates/fiml/tests/model_input_parity.rs)
and the Python contract test in
[`crates/fiml-python/tests/test_model_input_parity.py`](../crates/fiml-python/tests/test_model_input_parity.py).

Both runtimes verify active raw and model ordering, vector capacity and length,
reserved cells, warm-up `NaN` propagation, any-event clock updates, full-window
trade-price SMA state, standard scaling, and preservation of the SMA across a
time event. Rust also verifies that deserializing and reserializing the artifact
produces the same canonical JSON value. Test sensitivity was checked by changing
one expected day literal from `4` to `5`: both contract tests failed on the same
value mismatch before the literal was restored.

### P2: Allocation regression verification

Status: open

Priority: high

The compiled hot path contains no intentional allocation, but this requirement
is currently established by code inspection rather than a regression test.

Completion criteria:

- Add an isolated allocation-counting test or benchmark.
- Assert that accepted steady-state `Pipeline::handle_event` calls allocate
  zero times.
- Cover both identity and standard-scale operations.

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

The current tree passes:

- 173 Rust tests across all workspace targets and features;
- 74 Python tests;
- the maintained notebook;
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`.

## Deferred work

Do not add transformation chaining, a general transformation graph, runtime
state serialization, or speculative transformation variants. Add another
transformation only when a concrete model-training requirement cannot be
expressed by `Identity` or `StandardScale`.

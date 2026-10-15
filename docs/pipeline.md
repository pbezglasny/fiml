# Pipeline status

Status: in progress

Last updated: 2026-10-15

The pipeline runtime, model-input serialization, and Python interface are
implemented, including Python-fitted sklearn `SimpleImputer`, `StandardScaler`,
`RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`, `PowerTransformer`,
`VarianceThreshold`, and PCA
stages with Rust online inference.
Remaining interface-hardening items are listed below.

## Scalar definition migration (Rust)

`TransformerDefinition` stores public `input` and `output` IDs alongside a
public `TransformerOperation` enum containing only calculation parameters.
Existing constructors such as `TransformerDefinition::lagged(input, output, 2)`
are unchanged. Replace direct enum construction such as
`TransformerDefinition::Lagged { input, output, lag_window: 2 }` with:

```rust
use fiml::{FeatureId, TransformerDefinition, TransformerOperation};

let definition = TransformerDefinition {
    input: FeatureId::new("price"),
    output: FeatureId::new("previous_price"),
    operation: TransformerOperation::Lagged { lag_window: 2 },
};

if let TransformerOperation::Lagged { lag_window } = &definition.operation {
    println!("{}: {} events ago", definition.input.as_str(), lag_window);
}
```

Read IDs directly from the struct and match operation parameters through
`definition.operation`. `TransformerOperation::Identity` has no fields.
Pipeline construction still validates IDs and operation parameters. Python
builders and version `3.0` JSON retain their existing format and behavior;
saved artifacts need no migration.

## Rolling scalar statistics

`TransformerDefinition::rolling_std`, `rolling_min`, `rolling_max`, and
`rolling_range` consume any preceding-stage scalar, including fitted vector
outputs. Sample constructors take `(input, output, window, warmup_policy)`;
`window` is a positive observation count, without the event-lag 10,000 limit.
Use existing `sma` for rolling mean. Standard deviation is population std
(`ddof=0`); one observation has zero std and range.

The corresponding `_timed` constructors take a `Duration` window and default
to 1 ms aggregation. `_timed_with_aggregation` takes
`(input, output, window, aggregation, warmup_policy)`. Durations must be positive
whole milliseconds; windows must be multiples of aggregation. Every finite
observation contributes, including repeated timestamps and repeated values.
Epoch-aligned buckets expire when `bucket_start + window <= source_clock`.
Coarse aggregation therefore expires whole buckets; it does not discard spikes
inside them. This follows timed volatility's bucket semantics, rather than the
last-observation sampling used by timed lag/change transformers.

Only observed finite inputs enter history. An observed NaN or infinity emits
NaN until the next finite observation; history is preserved. Unobserved inputs
retain outputs, except source-clock advances refresh timed expiry and mark the
timed output observed for downstream stages. Context replacement does not
advance or refresh timed statistics, and context-only inputs have no clock.
Composite clocks follow existing vector-stage rules; every observed finite
value at that clock contributes. Empty windows and nonfinite arithmetic results
emit NaN. Extremely large finite inputs can overflow the variance calculation;
those summaries stop affecting results when their samples or buckets expire.

`FirstValue` exposes the current nonempty window immediately. `FullWindow`
requires the configured number of finite observations for samples, or elapsed
time since the first finite observation for timed windows. Readiness persists
through gaps, but an empty timed window still emits NaN.

All statistics and warmup policies for one input share circular history within
a stage; timed groups also share aggregation. Each distinct window is queried
once. A preallocated summary tree merges shifted moments and exact min/max,
without subtractive variance updates or allocation during event processing.
Updates and distinct-window queries cost `O(log slots)`. On 64-bit targets the
tree uses `96 * next_power_of_two(slots)` bytes, plus `8 * slots` for timed bucket
IDs. Sample slots equal the largest window. Timed slots equal
`max_window / aggregation + 2`, capped at 1,000,000 (about 104 MiB including the
tree at that cap). Choose coarser aggregation to reduce memory for long windows.

Version `3.0` JSON adds `rolling_std`, `rolling_min`, `rolling_max`,
`rolling_range`, and their `_timed` forms. All require `input`, `output`, `window`,
and `warmup_policy`; timed windows use duration strings and optional
`aggregation` (default `"1ms"`, omitted when serialized at its default).
Existing artifacts retain their behavior; older runtimes cannot read these new
types. See the [schema](pipeline-spec.schema.json) and the
[multi-scale market-state example](../crates/fiml/examples/transformers.rs).

## Rolling trajectory shape

`rolling_slope`, `rolling_path_length`, `rolling_positive_movement`,
`rolling_negative_movement`, `rolling_direction_change_count`, and
`rolling_max_drawdown` use the same sample-window constructor signature as
rolling statistics. They consume any preceding-stage scalar, including other
scalar transformations and fitted model components. Windows must be positive
observation counts; these transformers have no timed variants.

For ordered values `x[0..n]`, slope is ordinary least-squares regression against
sample indices `0..n-1`, in input units per observation. Path length sums
`abs(dx)`; positive movement sums `max(dx, 0)`; negative movement sums
`max(-dx, 0)`. Thus path length equals positive plus negative movement, while
last minus first equals positive minus negative movement. Direction changes
count reversals after discarding zero deltas: `+, 0, -` is one reversal.
Only deltas whose endpoints are inside the window contribute. Maximum drawdown
is `max(0, max(x[i] - x[j]))` for `i < j`, in absolute input units; it also works
for zero or negative inputs and is not a percentage.

| Window | Slope | Path length | Positive | Negative | Reversals | Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[1, 2, 3, 4]` | 1 | 3 | 3 | 0 | 0 | 0 |
| `[1, 4, 1, 4]` | 0.6 | 9 | 6 | 3 | 2 | 3 |
| `[1, 3, 3, 1]` | 0 | 4 | 2 | 2 | 1 | 2 |

The first two paths have identical endpoints and means but different shapes.
`FullWindow` waits for `window` finite observations; `FirstValue` exposes partial
windows. Slope always needs at least two observations, so `window=1` always
produces NaN for slope. Other singleton measurements are zero. Observed NaN or
infinity emits NaN without entering history; the next finite input connects to
the previous retained finite value. Unobserved inputs retain their outputs.
Nonfinite arithmetic results emit NaN and recover when the affected samples
leave the window.

All six kinds share one trajectory tree per input and stage across windows and
warmup policies, separate from the smaller statistics tree. Each distinct
window is queried once. Updates and queries cost `O(log max_window)` and make
no heap allocations. On 64-bit targets, tree storage is
`192 * next_power_of_two(max_window)` bytes (about 24 MiB for 100,000 samples).
Regression moments use an observed origin to preserve small changes in large
values. Tree merges preserve chronological order through circular wraparound.

Version `3.0` JSON uses the six names above as `type`, with required `input`,
`output`, `window`, and `warmup_policy` fields. Existing artifacts retain their
behavior; older runtimes cannot read these new types.

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
`FittedStage::Scalar` selects, renames, scales, averages, lags, computes changes, or measures rolling statistics of columns from the preceding
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

`TransformerDefinition::delta`, `simple_return`, and `log_return` use exact
finite-observation lags instead of accepted-event lags. A lag `k` in `1..=10_000`
requires `k + 1` finite observed inputs. Equal-valued observations count;
unobserved events retain the last output. An observed nonfinite input emits NaN
without advancing history. Each output inherits its input's observation flag,
including during warm-up and invalid arithmetic domains.

Delta computes `current - previous`; simple return computes
`current / previous - 1` (NaN for a zero denominator); log return computes
`ln(current) - ln(previous)` (NaN unless both values are positive). Finite zero
and negative values still enter history. Delta/simple-return arithmetic can
overflow; subtracting logs avoids intermediate ratio overflow for log returns.
One ring buffer sized to the maximum lag serves all change kinds and windows
for one input in one stage, with no allocations during warm-up or updates.
The additive JSON types `delta`, `simple_return`, and `log_return` have
`input`, `output`, and `lag_window` fields and retain pipeline version `3.0`.
See the [order-book spread example](../crates/fiml/examples/transformers.rs)
and [Python builders](../crates/fiml-python/README.md#fitted-model-input-pipelines).

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

- `just test` (Rust tests, Python tests, and the maintained notebook);
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
slots never set observation flags, so SMA/EMA and change transformers do not sample context replacements.
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

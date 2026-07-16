# ADR 0002: Unify indicator definition and compilation

Status: accepted  
Date: 2026-07-16

## Context

Built-in features currently have two parallel construction models:

- `FeatureSet` describes one output cell per `FeatureDef`.
- `IndicatorFeatureVectorBuilder` describes grouped indicators that may write
  several output cells.

Validation, generated names, event routing, window collection, output-index
wiring, and runtime construction are consequently repeated across feature
modules and builders. Adding an indicator requires coordinated edits to
several parallel matches and construction paths.

The runtime representation is already optimized around one indicator instance
serving several windows. The declarative model should express the same concept
instead of translating between one-definition-per-cell and
one-indicator-per-group models.

The library is pre-release. Breaking configuration, JSON, Rust, and Python
interfaces is acceptable when it produces a smaller and clearer architecture.
Calculation and event-dispatch hot paths must remain allocation-free, but
configuration and compilation may allocate.

## Decision

Use one grouped indicator definition model and one compiler for every
construction interface.

The construction flow is:

```text
FeatureSet or fluent builder
        |
        v
ordered IndicatorDef values
        |
        v
validation and compilation
        |
        v
fixed-capacity IndicatorFeatureVector
```

The compiler is an implementation detail. The low-level public compilation
interface is:

```rust
IndicatorFeatureVector::<F, V, M>::from_feature_set(cells, &feature_set)
```

The runtime-sized `FeatureExtractor` selects a fixed capacity and delegates to
the same interface. The fluent builder produces a reusable `FeatureSet`; it
does not own output storage or maintain a separate pending-feature hierarchy.

No compatibility shims will be retained for the old builders or JSON schema.

## Definition model

`FeatureSet` is an ordered collection of `IndicatorDef` values.

An `IndicatorDef` contains:

- a symbol name for symbol-scoped indicators;
- one `IndicatorSpec`;
- one or more ordered output definitions encoded by that specification.

One definition represents one runtime indicator instance. Its outputs occupy
one contiguous `OutputSpan`, represented by a start index and count. Output
`i` writes to `span.start + i`; runtime adapters do not store arrays of
arbitrary output indexes.

Indicator definitions and their outputs remain in caller-defined order. The
compiler must not reorder output cells. Compatible definitions are not merged
automatically: defining the same indicator identity more than once is an
error, and callers must place its windows in one grouped definition.

Indicator identity includes:

- symbol, when the indicator is symbol-scoped;
- indicator type;
- input source, when configurable;
- aggregation duration for bucketed indicators.

Output windows are not part of indicator identity.

## Output cardinality

Grouped SMA, EMA, timed SMA, and timed OBV definitions contain ordered window
lists. A single output uses a one-element list.

Each grouped definition must:

- contain at least one output;
- contain no duplicate window length;
- contain no more than `MAX_OUTPUTS_PER_INDICATOR`, initially 16.

Timed trade count remains a single-window, single-output indicator in this
change. Multi-window trade count is deferred.

The runtime `FeatureExtractor` remains limited to 128 output cells. Its
indicator-instance count is also limited to 128. Static compilation additionally
checks the caller-provided fixed capacities.

Output storage must have exactly the compiled output count. Unnamed or unused
trailing cells are not permitted, so feature names and values always have the
same length.

## Value sources

Moving averages use a dedicated `ValueSource` rather than the broader
`EventKind`. The initial sources are:

- `Price`;
- `Volume`;
- `TradePrice`;
- `TradeVolume`.

`ValueSource` owns the mapping from an event payload field to its dispatch
route. Unsupported event/source combinations are unrepresentable. Price is the
fluent and Python builder default, while serialized definitions store the
source explicitly.

OBV and trade count have fixed trade-event inputs and expose no configurable
source.

## Timed windows

Timed SMA and timed OBV share one `TimeWindows` definition containing:

```rust
TimeWindows {
    aggregation: Duration,
    windows: Vec<Duration>,
}
```

Compilation validates that:

- aggregation is at least one millisecond;
- every window is at least one aggregation;
- every window is an exact multiple of aggregation;
- all conversions and derived durations fit in `i64` milliseconds.

The compiler converts validated durations to bucket periods before runtime
construction. Timed trade count uses the same validation rules for its single
window.

## Generated feature names

Arbitrary user-provided output names are removed. The library generates stable,
globally unique canonical names from structured feature identity.

Names use colon-separated segments containing the source symbol, input source,
indicator type, and output parameters, for example:

```text
AAPL:price:sma:20
AAPL:trade_price:ema:10
AAPL:price:sma_timed:1000ms:60000ms
AAPL:trade:obv_timed:1000ms:60000ms
AAPL:trade:count_timed:1000ms:60000ms
clock:day_of_week
```

The symbol segment escapes `%` and `:` so separators remain unambiguous.
Compilation checks generated names for global uniqueness.

Aliases are not included. They may be added later if a concrete use case
justifies the additional interface and metadata.

Because names are global and include the symbol, lookup becomes:

```rust
fn index_of(&self, canonical_name: &str) -> Option<usize>;
```

The core stores generated names once and exposes them by borrowed slice rather
than cloning them on every call. Uncompiled `FeatureSet` values do not expose
feature names. They expose explicit `indicator_count()` and `output_count()`
metadata instead of an ambiguous `len()`.

## Clock features

Current clock features update from every event and do not filter by symbol.
They are therefore modeled as global definitions without a symbol.

`DayOfWeek` remains a global clock feature.

`TimeSinceSessionOpen` is renamed to `TimeSinceFirstEventOfDay`. Its current
behavior is elapsed time since the first observed event after a local day
boundary; it does not model an exchange calendar or configured session open.

Its fixed UTC offset:

- must be within `-14h..=+14h`;
- must use whole-minute precision.

A future calendar-aware or symbol-specific session feature must be designed as
a separate indicator with explicit filtering and calendar semantics.

## Validation and errors

The compiler is the single semantic validation path for `FeatureSet`
construction. Fluent builders collect definitions and delegate validation to
compilation.

Validation is fail-fast. Invalid definitions return:

```rust
FimlError::InvalidIndicatorDefinition {
    index: usize,
    reason: String,
}
```

The reason includes the indicator type, symbol when present, offending field,
and value. Capacity and timestamp errors remain separate error variants.

Standalone low-level indicator types remain public and validate their own
invariants for direct callers. The compiler delegates construction to those
types and adds definition context to errors.

Compilation may use temporary `Vec` storage to validate and fully construct
entries before moving them into fixed-capacity runtime arrays. This avoids
partially initialized runtime arrays and simplifies failure cleanup.

## Runtime and allocation contract

Configuration, JSON processing, canonical-name generation, and compilation may
allocate.

The compiled extractor owns and preallocates all storage needed to:

- validate event ordering;
- route an event;
- update indicator state;
- write every output cell.

Event dispatch and feature-value updates perform no per-event allocation.

Fixed-capacity runtime feature arrays remain in this change. Replacing them
with boxed slices is deferred until dispatch benchmarks demonstrate that the
simpler representation meets performance goals.

Runtime adapters remain concrete (`SmaFeature`, `EmaFeature`,
`ObvTimedFeature`, and so on). Small internal primitives such as `OutputSpan`
and an inlined output-writing helper may be shared. A universal generic feature
wrapper will not be introduced.

## Event ordering

The core extractor owns one global nondecreasing timestamp watermark.
Every dispatched event, including an event not consumed by a symbol-specific
indicator, must have a timestamp greater than or equal to the previous
successfully dispatched event.

Python delegates ordering validation to the core instead of maintaining a
second global rule while the core maintains per-stream `HashMap` state. This
removes differing semantics, duplicated validation, and potential hot-path map
allocation.

## Module ownership

The feature layer is organized by responsibility:

- `features/definition.rs` owns `FeatureSet`, `IndicatorDef`, `IndicatorSpec`,
  `ValueSource`, `TimeWindows`, and canonical-name definitions.
- `features/builder.rs` owns fluent construction of definitions.
- `features/compiler.rs` owns validation, output spans, routing metadata,
  canonical names, and runtime construction.
- `features/indicator_vector.rs` owns compiled storage and dispatch.
- `features/builtin/` owns concrete runtime event-to-indicator adapters.
- `indicators/` owns calculation algorithms only.

Indicator calculation modules do not know feature names, symbols, event
routing, or output cells.

## Numerical behavior

This architectural change preserves existing numerical indicator behavior.
In particular, it does not change sample-SMA warm-up behavior even though that
behavior and current warm-up documentation need a separate decision.

## Implementation sequence

1. Add extractor-dispatch benchmarks covering one output, grouped outputs,
   multiple symbols/routes, and clock dispatch.
2. Add characterization tests for existing numerical behavior.
3. Introduce the grouped definition types, canonical naming, `ValueSource`,
   `TimeWindows`, and `OutputSpan`.
4. Introduce the single compiler using temporary initialized storage.
5. Migrate SMA and EMA, then timed SMA and OBV, then single-window trade count
   and global clock features.
6. Replace the parallel fluent builder implementation with a `FeatureSet`
   builder.
7. Move global ordering into the core and remove Python ordering duplication.
8. Update Rust examples, Python bindings/tests, notebooks, JSON fixtures, and
   documentation.
9. Consolidate ring-buffer contract tests and repeated floating-point test
   helpers after production architecture stabilizes.

Each migration replaces and deletes its old construction path in the same
change. The repository must compile and pass tests between migrations; a
long-lived parallel compiler will not be maintained.

## Consequences

- Adding an indicator has one definition and compilation path.
- Public definitions match the optimized runtime grouping model.
- Output ordering and adjacency become explicit invariants.
- Moving-average input fields become readable and type-safe.
- Generated names and lookup are deterministic across Rust and Python.
- Dispatch loses per-stream timestamp-map bookkeeping.
- Construction becomes easier to reason about and safer on errors.
- Public Rust, Python, and JSON interfaces break.
- Builder and compilation paths may allocate, while runtime dispatch remains
  allocation-free.
- Fixed-capacity runtime complexity remains until benchmark evidence supports
  changing it.

## Deferred decisions

- Make timed trade count multi-window and share one bucket buffer across its
  outputs.
- Define sample-SMA warm-up semantics and align documentation and model-quality
  tests.
- Compare fixed-capacity runtime arrays with boxed-slice storage using the new
  dispatch benchmarks.
- Add feature aliases only when a concrete use case demonstrates that canonical
  generated names are insufficient.

# Refactor status

Status as of 2026-08-28.

## Current architecture

The feature-extraction path has one compilation flow:

```text
FeatureDefinition / FeatureKey
    -> FeatureExtractorBuilder or FeatureVectorSpec
    -> feature compiler
    -> FeatureExtractor
```

Each `FeatureDefinition` describes one scalar output cell. During compilation,
compatible definitions, such as moving averages with several windows, are
grouped into one runtime derivation with a contiguous output span.

`FeatureVectorSpec` is the versioned configuration and serialization boundary.
It owns canonically ordered scalar definitions, the complete model width, and
optional checksum metadata. With the `serde` Cargo feature enabled, it maps to
and from the strict grouped JSON format without exposing wire-only types in the
public API.

`FeatureExtractor<F, V>` owns:

- the caller-selected `V: FeatureVector` output storage;
- statically dispatched `FeatureDerivation` values;
- output spans and stable feature IDs;
- the symbol/event router;
- the latest accepted global event timestamp.

Event handling does not allocate. Routing uses precompiled arrays and flattened
subscriber lists, and derivations write directly into the output vector.

## Completed work

- Replaced the legacy grouped `IndicatorSpec`, `ScopedIndicator`, `TimeWindows`,
  and `ValueSource` core model with scalar `FeatureDefinition`, `FeatureKey`,
  `FeatureSource`, and `EventField` values.
- Rebuilt `FeatureVectorSpec` as a versioned, canonically ordered configuration
  that compiles through the same feature compiler as the fluent Rust builder.
- Added strict `FeatureVectorSpec` serialization behind the optional `serde`
  feature. The adapter supports all current feature keys, validates the wire
  contract, and preserves capacity and checksum metadata.
- Made the feature compiler the only path from definitions to runtime
  derivations and routing state.
- Replaced the old fixed-capacity indicator-vector and dynamic-extractor split
  with the generic `FeatureExtractor<F, V>`.
- Added runtime-sized `VecFeatureVector` support alongside
  `ArrayFeatureVector`.
- Migrated the Python binding to the new spec, compiler, and extractor. Its
  fluent Python `FeatureVectorSpec` expands grouped calls into scalar core
  definitions while retaining the established Python column names and
  canonical ordering.
- Preserved Python batch validation, transactional replay, output dtype
  selection, and `NaN` initialization for unavailable or reserved output cells.
- Restored the pipeline and transformation source files. Transformations are
  compiled, but the legacy pipeline remains excluded from `features/mod.rs`
  until its API and parity requirements are decided.
- Added distinct order-book delta and snapshot event kinds to the Rust event
  model, including constructors and routing support.
- Removed the legacy extractor, obsolete examples, and stale benchmark.
- Updated the maintained Rust examples to use `FeatureExtractor`, added a JSON
  `FeatureVectorSpec` example, and registered them as explicit Cargo targets.
- Added a deterministic historical trade replay example with a checked-in CSV
  fixture. It exercises trade-price SMA/EMA, trade-volume SMA, CVD, and timed
  trade count, emits feature columns from `feature_ids()`, and verifies schema,
  warm-up `NaN` values, ordering, row count, and final feature values.
- Updated `docs/project-schema.md` for the current architecture.

## Available feature derivations

The compiler currently supports:

- simple moving average;
- exponential moving average;
- cumulative volume delta;
- timed simple moving average;
- timed on-balance volume;
- timed trade count;
- day of week;
- time since the first event of the local day.

Moving averages can consume price, volume, trade-price, or trade-volume event
fields. Trade-based indicators subscribe to complete trade events. Timed and
clock derivations receive every accepted event so their windows advance even
when an event does not provide a matching sample.

## Verification

As of 2026-08-28, the current Rust workspace passes:

- 159 core unit tests and 2 historical replay example tests;
- compilation and execution of all workspace test targets, benchmarks, and
  registered examples with all Cargo features enabled;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`.

The most recent Python-specific verification, on 2026-08-24, also passed 52
Python tests and the maintained end-to-end notebook.

The project remains in pre-release development, so breaking API changes are
still expected. The extractor refactor, pipeline source restoration, and
feature-vector spec serialization are committed.

## Known gaps and deliberate boundaries

- The retained pipeline still targets the removed `IndicatorFeatures` and
  `IndicatorFeatureVector` APIs and is not declared by `features/mod.rs`.
  Adapting it mechanically would not resolve how transformations are
  serialized or kept identical between Rust and Python.
- `StandardScaler` and `ParallelTransformer` are Rust-only. Transformations are
  not represented in `FeatureVectorSpec`, serialized into the model artifact,
  or exposed through the Python binding.
- The library does not provide an exchange-specific historical-data loader.
- Label generation and model training remain outside the core API.
- Events from multiple live sources must be merged into globally nondecreasing
  timestamp order before they are passed to the extractor.
- The Rust event model accepts both order-book deltas and complete snapshots,
  but order-book state is not connected to feature derivations. The Python
  binding currently accepts order-book deltas only.
- The extractor intentionally compiles the closed set of library-owned
  `FeatureKey` and `FeatureDerivation` variants. User-defined derivations are
  not supported by the compiled extractor.

## Recommended next implementation

Decide whether transformations are part of the versioned cross-language model
artifact before exposing or adapting `Pipeline`. This decision must define how
transformation configuration is serialized and how Rust and Python preserve
identical calculation order and output layouts.

Order-book derivations should follow as a separate design task because they
must define ownership of order-book state and conversion from `Decimal` book
values to the extractor's `Float` output type.

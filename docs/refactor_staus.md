# Refactor status

Status as of 2026-08-24.

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

As of 2026-08-24, the current tree passes:

- 141 Rust unit tests;
- 52 Python tests;
- the maintained end-to-end notebook;
- compilation and execution of all workspace test targets, benchmarks, and
  registered examples with all Cargo features enabled;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`.

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
- There is no deterministic Rust historical-trade replay example yet. The
  library also does not provide an exchange-specific historical-data loader.
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

The next bounded implementation should be a deterministic historical trade
replay. It would exercise the complete public Rust path without adding an
exchange client or model-training concerns to the core library:

1. Read a checked-in, timestamp-sorted trade CSV containing timestamp, symbol,
   price, quantity, and aggressor side.
2. Construct scalar definitions for trade-price SMA/EMA, trade-volume SMA,
   CVD, and timed trade count.
3. Build one `FeatureExtractor` with output storage sized to the number of
   definitions.
4. Convert each input row to `Event::Trade` and call `handle_event`.
5. Write raw fields and current feature-vector values to an output CSV, using
   `feature_ids()` as the feature-column names.
6. Add a deterministic test covering the output schema, warm-up `NaN` values,
   row count, and final feature values.

`crates/fiml/examples/binance_trades.rs` remains the live-serving example. It
connects to the Binance trade WebSocket, converts messages to `Event::Trade`,
updates trade-price EMA and SMA derivations, and prints CSV rows. The historical
replay should become the primary end-to-end example because it is repeatable
and suitable for automated verification.

After the replay example, decide whether transformations are part of the
versioned cross-language model artifact before exposing or adapting `Pipeline`.
Order-book derivations should follow as a separate design task because they
must define ownership of order-book state and conversion from `Decimal` book
values to the extractor's `Float` output type.

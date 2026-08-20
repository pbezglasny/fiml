# Refactor status

Status as of 2026-08-20.

## Current architecture

The feature-extraction path now has one construction flow:

```text
FeatureDefinition / FeatureKey
    -> FeatureExtractorBuilder
    -> feature compiler
    -> FeatureExtractor
```

Each `FeatureDefinition` describes one scalar output cell. During compilation,
compatible definitions, such as moving averages with several windows, are
grouped into one runtime derivation with a contiguous output span.

`FeatureExtractor<F, V>` owns:

- the caller-selected `V: FeatureVector` output storage;
- statically dispatched `FeatureDerivation` values;
- output spans and stable feature IDs;
- the symbol/event router;
- the latest accepted event timestamp.

Event handling does not allocate. Routing uses precompiled arrays and flattened
subscriber lists, and derivations write directly into the output vector.

## Completed work

- Replaced the legacy `FeatureSet`, `IndicatorSpec`, `ScopedIndicator`,
  `TimeWindows`, and `ValueSource` core model with scalar `FeatureDefinition`,
  `FeatureKey`, `FeatureSource`, and `EventField` values.
- Made the feature compiler the only path from definitions to runtime
  derivations and routing state.
- Replaced the old fixed-capacity indicator-vector and dynamic-extractor split
  with the generic `FeatureExtractor<F, V>`.
- Added runtime-sized `VecFeatureVector` support alongside
  `ArrayFeatureVector`.
- Migrated the Python binding to the new compiler and extractor. Its fluent
  Python `FeatureSet` expands grouped calls into scalar core definitions while
  retaining the established Python column names and canonical ordering.
- Preserved Python batch validation and `NaN` initialization for unavailable
  output cells.
- Removed the old pipeline, feature-set serialization implementation, legacy
  extractor, obsolete examples, stale benchmark, and unused
  `ParallelTransformer` prototype.
- Updated the maintained Rust examples to use `FeatureExtractor` and registered
  them as explicit Cargo example targets.
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

The current refactor passes:

- 129 Rust unit tests;
- 29 Python tests;
- compilation of all workspace targets, benchmarks, and registered examples;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo fmt` and `git diff --check`.

The project remains in pre-release development, so breaking API changes are
still expected. The current refactor changes are not yet committed.

## Known gaps

- Feature definitions do not currently have a serialization format.
- The library does not provide an exchange-specific historical-data loader.
- Label generation and model training are outside the current core API.
- Events from multiple live sources must be merged into globally
  nondecreasing timestamp order before they are passed to the extractor.
- Order-book state exists, but complete order-book feature extraction and the
  order-book snapshot event path are not finished.
- Only the existing closed set of `FeatureKey` and `FeatureDerivation` variants
  can be compiled; user-defined derivations are not yet supported.

## Real-data end-to-end example

`crates/fiml/examples/binance_trades.rs` already demonstrates a partial
real-data path. It connects to the Binance trade WebSocket, converts messages
to `Event::Trade`, updates trade-price EMA and SMA derivations, and prints CSV
rows.

The recommended complete and reproducible example is a historical trade
replay:

1. Read a local trade CSV containing timestamp, symbol, price, quantity, and
   aggressor side.
2. Construct scalar definitions for trade-price SMA/EMA, trade-volume SMA,
   CVD, and timed trade count.
3. Build one `FeatureExtractor` with output storage sized to the number of
   definitions.
4. Convert each input row to `Event::Trade` and call `handle_event`.
5. Write raw fields and the current feature-vector values to an output CSV,
   using `feature_ids()` as the feature-column names.
6. Optionally load the generated CSV in Python to create labels and train an
   sklearn, LightGBM, XGBoost, or CatBoost model.

A historical replay should be the primary E2E example because it is
deterministic, repeatable, and suitable for tests. The existing WebSocket
example can then demonstrate live serving with the same definitions.

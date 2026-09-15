# Public Rust API audit for v0.1.0

Date: 2026-09-15  
Audited commit: `b637bab`  
Status: boundary accepted; all four findings resolved.

The extractor hides its compiler, routing tables, derivations, and transformer
runtime. Standalone indicators remain public and support explicit event time.

## Resolved findings

1. **Trade counter updates:** `TradeCountTimed::update(timestamp)` is now
   public, so downstream callers can construct, populate, and read a counter.
2. **Timed updates and expiry:** OBV and VWAP now accept
   `update(price, volume, timestamp)`; rolling trade volume accepts
   `update(volume, timestamp)`. These replace the system-clock wrappers.
   All six timed indicator types expose `observe(timestamp)` for expiry and
   warm-up without adding data. The extractor and batch helpers call the same
   timestamped update methods. Public docs describe the finite-input and
   nondecreasing-timestamp preconditions; reads do not advance time.
3. **Unused order-book export:** `SyncState` is private to `book.rs` and is no
   longer re-exported. Update outcomes and errors remain public.
4. **Unreachable declarations:** `EmaWindow` and `SmaWindow` are private.
   `LaggedFeature` is `pub(crate)` because it appears in the crate-visible
   `Transformer` enum; its constructor and `apply` method are `pub(super)`.
   None of these implementation types is exported to downstream users.

The changed timestamped update signatures and removal of `SyncState` are
intentional breaking changes before the first release.

## Accepted public boundary

| Surface | Decision |
| --- | --- |
| Events, event payloads, symbols, warm-up policy | Keep public: callers construct input and configure behavior. |
| Feature keys, IDs, definitions, builder, extractor, specs | Keep public: these are the supported configuration and execution paths. |
| Pipeline, pipeline spec, transformer definitions, fitted stages | Keep public: callers supply learned state and consume model inputs. |
| FeatureVector, array and heap output storage | Keep public: caller-selected storage is an intended extension point. |
| Standalone indicators and batch helpers | Keep public with explicit event time. This preserves the intent of ADR 0007. |
| RingBuffer, stack/heap buffers, constructors, bucket types | Keep public: standalone indicator signatures expose these storage types and bounds. Bucket fields stay private. |
| Order-book configuration, updates, queries, results and errors | Keep public; SyncState remains internal. |
| Public errors and their supporting enums | Keep public: callers need to inspect failures. |
| MAX_OUTPUTS_PER_INDICATOR and symbols::MAX_SYMBOL_NUMBER | Keep public: they describe enforced configuration limits. |
| Compiler, derivations, routing, prepared book updates, runtime stages and transformers | Keep internal; the current boundary already does this. |

Keep the existing `event`, `features`, `indicators`, `order_book`, and `symbols`
module paths and root re-exports. Removing duplicate import paths would add
churn without hiding additional implementation state.

## Verification

The downstream integration test in
[`standalone_timed_indicators.rs`](../crates/fiml/tests/standalone_timed_indicators.rs)
constructs all six timed indicators through the exported API and verifies
historical replay, warm-up completed by time advancement, an update at an
already observed timestamp, and repeated expiry without new samples.

All checks passed after the implementation:

- Formatting: `cargo fmt --all -- --check`.
- Workspace tests: `cargo test --workspace --all-features --offline`
  (254 tests and 1 doctest).
- Workspace Clippy: `cargo clippy --workspace --all-targets --all-features --offline -- -D warnings`.
- Strict docs: `RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc -p fiml --all-features --no-deps --offline`.
- Public visibility: `cargo rustc -p fiml --lib --all-features --offline -- -D unreachable_pub`.

The audit covered public signatures, re-exports, and uses in the Python
binding, examples, integration tests, benchmarks, and existing design records.
It was not a full numerical-correctness or performance audit.

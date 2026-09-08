# ADR 0009: Derive order-book features

Status: accepted

Date: 2026-09-08

## Context

`FeatureExtractor` already owns caller-configured order books and routes visible
book updates through `FeatureRoute::OrderBook`. The private
`FeatureDerivation::update_order_book` hook has no concrete implementations.
`OrderBook` already calculates the indicators needed for this first integration.

Order-book indicators should write into the same feature vector as existing
features, with stable IDs and no additional allocation during derivation. The
Rust builder accepts books through `add_order_book`, while the JSON/spec and
Python construction paths do not yet configure this state.

## Decision

Add six public `FeatureKey` variants: `OrderBookMidPrice`, `OrderBookSpread`,
`OrderBookSpreadBps`, `OrderBookWeightedMidPrice`, `OrderBookMicroprice`, and
`OrderBookImbalance`. Each takes a `symbol`; imbalance additionally takes
`n_levels: usize`. These features need neither an event source parameter nor a
warm-up policy.

Generate deterministic feature IDs using the existing symbol encoding,
`source=order_book`, and `n_levels` for imbalance. Preserve caller-supplied IDs.
Add corresponding indicator error labels and compiler support. Reject zero
imbalance depth, duplicate keys or IDs, and missing configured books during
extractor construction. Missing books return `OrderBookNotConfigured`.

Group imbalance definitions by symbol into one derivation, preserving their
relative output order and the compiler's existing grouped layout rules. Apply
the existing limit of 16 outputs per derivation. Store depths in fixed-capacity
storage and evaluate each configured depth with `OrderBook::imbalance`; no new
shared depth-calculation algorithm is introduced.

Implement the adapters in one private order-book derivation module and connect
them to the existing book route and update hook. Reuse the six existing
`OrderBook` calculations without changing their formulas. Convert their
`Decimal` results using `ToPrimitive::to_f64` from the installed dependency and
write directly into the assigned output span. Missing results or failed
conversions produce `NaN`.

All outputs start as `NaN`, including when the caller supplies a populated book.
Refresh book features after an applied update or resynchronization, using the
final visible book state. Buffered, ignored, rejected, and unrelated events
leave these outputs unchanged. In particular, a rejected sequence-gap update
does not clear the previous feature values. One grouped imbalance derivation
counts as one updated feature, matching existing runtime counting.

This first implementation supports the Rust feature builder. JSON wire support,
spec book configuration, and Python APIs are deferred. Extend exhaustive key
handling and canonical ordering so existing spec functionality continues to
work. Serialization of the new keys returns an explicit unsupported-feature
error; building a spec-based extractor without its required books returns the
missing-book error.

## Consequences

- Order-book values become ordinary named feature-vector outputs through the
  existing Rust builder and closed runtime dispatch.
- Derivation adds no heap allocation or dependencies. Book storage retains its
  existing allocation behavior; this decision does not make all book mutations
  allocation-free.
- Grouped imbalance reuses the existing calculation once per depth, so
  overlapping depth scans are repeated. A shared traversal can be considered if
  measurements show this work matters.
- Existing indicator formulas, available-depth behavior, and book
  synchronization semantics remain authoritative. No best-quote features,
  depth-query features, smoothing, or new indicators are added.
- Rust examples must demonstrate book configuration, feature definitions, and
  snapshot/delta updates. New types and the derivation module require intent
  documentation.
- Unlike the complete integration described in ADR 0007, serialization and
  Python support are deliberately deferred for these new features. Existing
  serialized features retain their current behavior.

## Validation

- Verify all six outputs against known book values, including distinct weighted
  mid-price and microprice results, empty and one-sided books, and requested
  depths larger than the available book.
- Verify grouped depth ordering, the 16-output limit, multiple-symbol routing,
  duplicate rejection, zero-depth rejection, missing-book rejection, stable
  default IDs, and custom IDs.
- Exercise applied snapshots and deltas, buffering, stale updates, sequence-gap
  rejection, and resynchronization with actual feature subscribers. Check
  initial `NaN` values and unchanged outputs for unrelated events.
- Check explicit serialization rejection for new keys and regression coverage
  for existing spec behavior.
- Reuse the existing allocation counter with preconstructed updates to existing
  levels to verify that feature derivation adds no allocation.
- Run formatting, workspace tests with all features, strict Clippy, and existing
  Python/notebook CI checks when implementing the runtime changes.

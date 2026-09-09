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
write directly into the assigned output range. Missing results or failed
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
- Check serialization round trips for new keys and regression coverage for
  existing spec behavior, as specified in the follow-up below.
- Reuse the existing allocation counter with preconstructed updates to existing
  levels to verify that feature derivation adds no allocation.
- Run formatting, workspace tests with all features, strict Clippy, and existing
  Python/notebook CI checks when implementing the runtime changes.

## Follow-up: serialize feature definitions

On 2026-09-08, JSON support was added to document order-book features in
`docs/example_of_store_definition.json`. This supersedes the serialization
deferral above. Book configuration and Python event ingestion remain deferred;
deserialized Rust definitions can be passed to the existing builder alongside
`add_order_book`. Spec-based extractor construction still rejects missing books.

The existing version `1.0` format gains six kinds matching the snake-case feature
key names, from `order_book_mid_price` through `order_book_imbalance`. Their source
is exactly `{"type": "order_book"}` and their symbol scope must be non-global.
They have no warm-up policy or calculation options. Scalar indicators have one
output with an optional custom ID; default scalar outputs may be omitted.
Imbalance requires one to sixteen outputs, each with a positive integer
`n_levels` and optional ID. Depth order is preserved. `window`, event/field source
parameters, and `n_levels` on other indicators are rejected. Book state and its
update policy are not serialized.

Both serialization and deserialization support these definitions. The JSON
schema, canonical pipeline example, and Rust/Python round-trip tests cover the
extension; existing feature representations and IDs remain unchanged.

## Follow-up: remaining quote and depth queries

The remaining `OrderBook` queries are exposed as scalar Rust `FeatureKey`
variants through the existing builder, compiler, route, and JSON format:

| Feature key suffix after `OrderBook` | Parameters besides `symbol` | Output |
| --- | --- | --- |
| `BestBidPrice`, `BestBidSize`, `BestAskPrice`, `BestAskSize` | none | Best quote price or size |
| `LevelSize` | `side`, `price` | Size at an exact price |
| `NthPrice`, `NthSize` | `side`, `n_levels` | Price or size at one-based depth |
| `DepthUntilPrice` | `side`, `price` | Cumulative size through the inclusive threshold |
| `DepthUntilSizePriceFrom` | `side`, `size` | First price needed to reach the target |
| `DepthUntilSizePriceTo` | `side`, `size` | Last price needed to reach the target |
| `DepthUntilSizeTotalSize` | `side`, `size` | Whole-level cumulative size reaching the target |
| `VolumeBetweenPrices` | `side`, `from_price`, `to_price` | Size in `[from_price, to_price)` on either side |

All queries reuse the current book methods. The three target-size outputs return
`NaN` when available depth cannot reach the target; the last level is included
in full, so total size can exceed the requested size. Missing quotes or levels
also return `NaN`. Cumulative depth and interval volume return zero when no
levels qualify. Depth indices and target sizes must be positive; query prices
must be nonnegative, and interval bounds must be increasing. These conditions
are validated before event handling. `top_n` is represented by selecting each
required price/size depth as a scalar feature, rather than a variable-width output.

New query definitions compile independently, with parameters included in their
identity. They retain existing initialization, symbol routing, buffering,
rejection, and resynchronization behavior. Queries allocate no memory during
derivation. Overlapping queries repeat book traversals; share scans only if
profiling warrants it. Existing grouped imbalance ordering and IDs are unchanged.

JSON kinds are the snake-case key names. Best-quote definitions use the existing
scalar representation. Parameterized queries require `options`, for example:

```json
{
  "kind": "order_book_depth_until_size_price_to",
  "source": {"type": "order_book"},
  "options": {"side": "bid", "size": "4"},
  "outputs": [{"id": "bid_price_for_four_units"}]
}
```

`side` is `bid` or `ask`; prices and sizes are exact decimal strings, parsed
without rounding. Serialization and default IDs normalize decimal scales.
`n_levels` is a positive integer in query options (imbalance retains its existing
per-output `n_levels`). Each query has one optional output ID and no warm-up
policy. Canonical query ordering uses kind and parameter-inclusive default ID;
custom IDs do not affect ordering. JSON schema checks parameter shape, while
Rust additionally validates decimal representability and increasing bounds.

The Rust example `cargo run -p fiml --example feature_extractor` demonstrates a
best-quote output and a target-size query. Python can round-trip these definitions;
book configuration and real Python event ingestion remain tracked in issue #112.

## Follow-up: configured Rust/Python replay (2026-09-09)

`FeatureExtractorSpec` owns an optional per-symbol `order_books` configuration
list. Each entry contains `symbol`, `update_policy` (`monotonic` or `contiguous`),
and `buffer_size`. All three fields are explicit. Rust exposes this through
`OrderBookConfig` and `FeatureExtractorSpec::with_order_books`; Python exposes
`FeatureExtractorSpec.configure_order_book`. Configuration is canonically sorted
by symbol. Duplicate symbols and the global symbol are rejected. Missing books
are rejected when building an extractor or pipeline. Existing version 1.0
artifacts without this optional list retain their behavior. Live levels, history,
sequence IDs, and synchronization state are never serialized. Every construction
starts with fresh books and `NaN` outputs. Zero buffer capacity retains the
existing core behavior, including capacity errors when retention is required.

Python exposes the book feature builders and immutable `OrderBookEvent.snapshot`
and `.delta` constructors. Events carry a symbol name, signed millisecond
timestamp, unsigned 64-bit update ID, and level tuples. Snapshot levels are
`(price, size)` pairs; delta changes are `(side, price, size)` triples, with side
`bid` or `ask`. Price and size inputs must be exact nonnegative decimal strings;
floats are rejected and values outside Decimal precision are rejected without
rounding. Zero sizes retain deletion semantics. `update_order_book(event)` and
`transform_order_book(events)` share the existing Rust dispatch for both raw
extractors and model-input pipelines. The obsolete numeric `KIND_ORDERBOOK`
placeholder now raises an explicit migration error instead of discarding quotes.

Book event constructors validate all payloads before exposing an event. Before
batch dispatch, `transform_order_book` checks every event's configured symbol and
timestamp ordering against both the runtime watermark and preceding rows. Any
such failure leaves the whole batch unapplied. State-dependent book errors are
handled sequentially: preceding rows remain applied, the rejected row performs
exactly the core synchronization transition (including retaining a gap delta),
and later rows are not applied. The error identifies the zero-based failing row;
callers may inspect current values and resynchronize with a snapshot. Raw/final
values and timestamp remain unchanged by a rejected row. Existing numeric/trade
batch prevalidation remains unchanged. Book payload construction and batch output
allocation occur at the Python boundary; derivation and transformation remain
allocation-free, with book storage retaining its existing allocation behavior.

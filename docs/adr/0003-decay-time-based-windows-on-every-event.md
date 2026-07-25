# ADR 0003: Decay time-based windows on every event

Status: accepted  
Date: 2026-07-25

## Context

Timed indicators (`SmaTimed`, `ObvTimed`, `TradeCountTimed`) expired their
buckets only inside `update_inner`, and their feature adapters called that only
when a trade for their own symbol arrived. Nothing advanced them on a `Time`
event or on another symbol's trade, so a window never decayed while its symbol
was quiet.

Observed on a compiled extractor carrying `trade_count_timed("BTC", aggregation
1s, window 2s)` and `day_of_week`:

```text
after 3 BTC trades   = [3.0, 4.0]
1 hour later (Time)  = [3.0, 4.0]
1 hour later (ETH)   = [3.0, 4.0]
```

"Trades in the last 2 seconds" still read 3 an hour after the last trade. In
live serving a quiet market reports its last burst indefinitely. In batch
extraction every row whose event belongs to another symbol carries a stale
as-of time. Because global clock features refresh on every event, one snapshot
mixed a live clock with frozen windows that disagreed about what "now" was.

This is not one of the decisions ADR-0002 deferred. It is a defect in the
dispatch model that ADR-0002 did not address, and it is independent of the
sample-SMA warm-up semantics that ADR-0002 explicitly left open.

## Decision

Every output cell is valid as of the last dispatched event's timestamp.

A compiled extractor advances every time-decaying feature on every dispatch,
whatever kind or symbol the event carries, before any new data is applied.

### Where decay happens

Decay happens inside `dispatch`, not at read time and not on a caller-driven
schedule. Dispatch order is:

```text
validate -> advance time-decaying features -> kind group -> every-event group
```

Validation comes first so a rejected event leaves no trace. Decay comes before
the update groups so expiry precedes insertion exactly once per event, and so
the groups observe an already-aged window.

A pull model computing values at read time would produce identical numbers under
the current one-snapshot-per-dispatch usage, and would be cheaper when many
events are dispatched between reads, but it makes reading `&mut` and turns a
forgotten refresh back into the stale value this ADR removes.

### The category, and how it is routed

A **time-decaying feature** is one whose value loses validity as the timestamp
advances, because its window is measured in time rather than in samples.
`IndicatorSpec::is_time_decaying()` names the category, beside the existing
`route()` and `is_global()`.

Compilation records the positions of these features and the compiled extractor
walks that list as a separate pass. `FeatureRoute` keeps its existing meaning —
which payload a feature consumes — and the every-event group keeps meaning
*global clock feature*, as `CONTEXT.md` defines it. Routing timed indicators
into that group would have been a smaller change but would have conflated
"subscribes to trades" with "needs a clock tick" and weakened a precise glossary
term.

### Empty windows

An empty window's value is decided per indicator, because the honest answer
differs:

- `TradeCountTimed` reports `0`. No trades in the last 60s is a true and
  informative fact about a quiet market.
- `ObvTimed` reports `0`. A signed-volume sum over no buckets is zero.
- `SmaTimed` reports `NaN`. The mean of no samples is undefined, and a `0.0`
  price average reads as a crash to zero — the fake-zero ambiguity that the NaN
  warm-up decision deliberately removed elsewhere.

A uniform NaN rule was rejected because it would discard the real signal in a
zero trade count.

### Decay never overwrites a warm-up cell

An indicator that has never received data is not advanced, so its cell keeps the
NaN a compiled extractor prefills. "No trades in the last 60s" is a claim about
an observed window; before a symbol's first event there is no observed window to
make it about, and writing `0` there would be indistinguishable from a genuinely
quiet market.

This keeps the change within its scope. Warm-up semantics stay exactly as they
were, so ADR-0002's deferred decision remains open and undisturbed. Concretely,
in a multi-symbol frame the cells of a symbol that has not yet traded stay NaN
while other symbols' events flow past, and start decaying normally from that
symbol's first event.

Each timed indicator answers this with `has_observations()`. Expiry advances a
cursor rather than dropping buckets, so it stays true once the first event
arrives, including after a window has decayed to empty.

### Contract

`Feature<F>` gains `advance_to(now, output)` with a default no-op body, so
carrying a clock dependency is part of the stated feature contract and costs
nothing for features that do not. `BuiltinFeature` overrides it for the three
timed variants.

Each timed indicator gains a pure `advance_to(now)` in `indicators/`, which
knows nothing about cells, symbols or routing, per ADR-0002's module ownership.
Its adapter decides when to call it and what to write. The adapters deliberately
do **not** filter by symbol when advancing: decay is driven by any event's
timestamp, which is the entire point.

`update_inner` calls `advance_to(now)` itself rather than relying on the refresh
pass, so the standalone public indicator types stay correct for direct callers.
Because advancing is idempotent, the second call costs one comparison.

## Properties

**Bucket-boundary and raw-timestamp expiry are equivalent.** `ObvTimed` and
`TradeCountTimed` expired against `bucket_start(now)` while `SmaTimed` expired
against raw `now`. Every bucket timestamp and every window duration is an exact
multiple of the aggregation, and `bucket_start(now)` is the largest such
multiple not exceeding `now`, so both thresholds classify every bucket
identically. All three now take `now`.

**Advancing is monotone and idempotent in the as-of timestamp.** Expiring at
t=30 and then at t=60 removes exactly the buckets that expiring at t=60 removes.
Extra events between two trades therefore add snapshot rows to a stream without
changing the values observed at the trades themselves, which is what keeps a
live Rust stream carrying heartbeats in parity with a heartbeat-free Python
replay of the same trades. Pinned by
`interleaved_time_events_do_not_change_values_at_trade_rows`.

**A bucket covering `now` can never be expired.** Its start is `bucket_start(now)`
and every window is at least one aggregation long, so it always remains inside
every window. This removed the branch in `SimpleMovingAverageTimed::update_inner`
that could resurrect an already-expired bucket: once expiry always runs first
against the same `now`, that branch became unreachable. It is now a
`debug_assert!` documenting the invariant.

## Scope

Decay only. ADR-0002's deferred decision on sample-SMA warm-up semantics remains
open and is unaffected: a partially filled time window keeps reporting the mean
of the samples inside it, which is correct by definition, unlike the sample SMA
dividing a partial sum by its full period.

## Numeric compatibility

`FEATURE_SET_FORMAT_VERSION` is unchanged. The artifact schema does not change;
the numbers computed from an unchanged artifact do. The library is pre-release,
AGENTS.md permits breaking changes until the first release, and no wheel has
been published, so no trained model exists against the previous semantics.

This exposes a real gap — schema compatibility and numeric compatibility share
one field — which is recorded for a future decision rather than resolved here.

## The `decimal` feature is removed

`Float` gains an unconditional `const NAN`, required by the empty-window rule.
`rust_decimal::Decimal` has no NaN representation, so it can no longer implement
`Float`.

Putting the sentinel on a separate capability trait was rejected: `BuiltinFeature`
contains `SmaTimedFeature`, so the extra bound would propagate through the enum,
`IndicatorFeatureVector` and nearly every generic signature in the crate — to
preserve an impl with no users, no tests, and no path to the feature layer, since
`FeatureExtractor` is hardcoded to `f64`.

The designed way to restore exact arithmetic is a local newtype:

```rust
pub struct Nullable<T>(Option<T>);
impl Float for Nullable<Decimal> { const NAN: Self = Nullable(None); /* ... */ }
```

Two constraints were established by prototyping before this ADR:

- The newtype is **required**. `impl Add for Option<Decimal>` is rejected by the
  orphan rule (E0117); neither `Option` nor `Decimal` is local to this crate.
- `PartialOrd` must be **hand-written**. `Option`'s derived ordering makes
  `None < Some(2)` true, which would silently corrupt `OnBalanceVolumeTimed::sign`.
  IEEE semantics require every comparison against missing to be false.

Two incidental gains if it is built: division by zero can map to missing rather
than panicking as `Decimal::div` does, and `NAN == NAN` becomes true, removing
the `equal_nan=True` awkwardness the Python README documents for parity checks.

The name `Nullable` is provisional and should be settled when exact arithmetic
is actually needed.

## Consequences

- A feature-vector snapshot has one consistent as-of time across every cell.
- Timed features decay to their empty value during quiet periods instead of
  reporting a frozen burst.
- Batch extraction over multi-symbol frames is corrected: rows belonging to one
  symbol now age every other symbol's windows.
- Python needs no API change. `compute_features` gains correct multi-symbol
  decay, and `KIND_TIME` rows already let callers force decay at a decision
  point.
- Every dispatch pays for the refresh pass. Measured on three time-decaying
  features: dispatch of an event that feeds them goes 24.6ns to 32.5ns, and of
  an event that does not, 8.4ns to 14.8ns. The pass is proportional to the
  number of time-decaying features, so it compounds the existing linear scan
  over indicators; that scan is a separate, still-open performance question.
- Feature values change for any stream with gaps, without an artifact change.
- Exact-decimal arithmetic is unavailable until `Nullable<T>` is built.

## Deferred decisions

- Skip the output write when nothing expired, if the refresh pass shows up in a
  profile.
- Key dispatch by symbol as well as kind, so per-event work is proportional to
  the matching indicators rather than to all of them.
- Version numeric semantics separately from the artifact schema.
- Build `Nullable<T>` when exact arithmetic has a concrete use case, and settle
  its name then.

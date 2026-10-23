# ADR 0004: Make window-indicator warm-up configurable

> Ordering update (2026-09-08): ordering and timed-window advancement are now
> per symbol across all event kinds. Global Time ticks affect only GLOBAL;
> global calendar features follow the maximum accepted timestamp. This
> supersedes the global-ordering/expiration language below.

Status: accepted  
Date: 2026-07-26

## Context

Window indicators exposed inconsistent values before their configured history
was available. Sample SMA emitted a partial mean, EMA emitted its seeded value,
and rolling cumulative indicators emitted partial sums. Timed indicators also
advanced only on matching events, so their readiness and values could remain
stale while global event time moved forward.

Feature cells need one missing-value contract for train/serve parity, while
callers may still need partial values for particular models. Readiness must be
explicit rather than inferred from NaN because a ready timed average can have no
samples in its current window.

ADR 0003 defined partial-window sample-SMA arithmetic but deliberately deferred
the broader warm-up contract. This ADR supersedes that output policy. The
underlying SMA calculation may still maintain a correct partial mean while the
feature output withholds it.

## Decision

Window indicators with partial-value semantics have one `WarmupPolicy`.
Exact sample-lag returns are the exception described below.

```rust
pub enum WarmupPolicy {
    FirstValue,
    FullWindow,
}
```

The policy applies to every output of a grouped indicator. Each output becomes
ready independently according to its configured window. Readiness is monotonic
and resets only when the indicator is reset or reconstructed.

`FirstValue` makes every configured output ready after the indicator's first
matching input. `FullWindow` uses these rules:

- sample SMA, EMA, and CVD become ready after the window period's number of
  matching inputs;
- timed SMA, timed OBV, and timed trade count become ready after global event
  time has advanced by the complete window duration from the first matching
  input.

Empty timed aggregation buckets count as observed time. Timed indicators observe
every dispatched event so readiness advances and expired buckets are removed
even without another matching market event.

A ready timed SMA with no samples in its current window has no current value.
Timed OBV and timed trade count use zero for an observed empty window. Readiness
therefore does not imply current-value availability.

Standalone indicators expose `is_ready_at(index)` for one output and
`is_ready()` for all configured outputs. Their `value_at(index)` methods return
`None` when an output is warming up, unavailable, or not configured.

Feature-vector cells are `f64`, initialized to NaN, and remain NaN while
`value_at` returns `None`. Decimal cannot represent this missing-value contract.

### Refinement: exact sample-lag returns (issue #102)

A positive lag `k` compares the current price with the price exactly `k`
matching samples earlier. Each output becomes ready after `k+1` samples;
shorter lags can become ready before longer ones. No lag is shortened during
warm-up, and no future sample is used. Neither `FirstValue` nor `FullWindow`
applies: standalone returns, feature keys, Python builders, and JSON accept no
warm-up policy. JSON containing a return `warmup_policy` is rejected.

History advances only for the configured symbol and scalar source. Standalone
price and trade-price inputs have separate histories; other symbols, volume
events, and time ticks leave those histories and outputs unchanged. Compatible
lags share one preallocated ring buffer retaining `max(lags)+1` values.

Simple returns express fractional change, `current / previous - 1`, calculated
as `(current - previous) / previous` in `f64`. Missing history or a denominator
of either `+0.0` or `-0.0` produces `None` in standalone Rust and NaN in feature
vectors. Finite negative prices use ordinary arithmetic, and a zero current
price with a nonzero denominator produces `-1`. Zero and negative samples still
advance history. Readiness remains true after sufficient history exists even
if the current denominator is zero; values recover when it becomes nonzero.
Existing logarithmic returns use the same readiness rule but require both
prices to be positive.

The return contract preserves existing API signatures, canonical feature IDs,
and extractor spec version `2.0`. Shared Rust/Python literal replay snapshots
cover the contract in `tests/fixtures/sample_returns.json`.

## Configuration and parity

For indicators supporting partial values, `WarmupPolicy` is stored in every
window `IndicatorSpec` and serialized as
`"first_value"` or `"full_window"`. It is not part of indicator identity or
canonical feature names.

Rust standalone indicator constructors require the policy explicitly. Rust
feature-vector spec builder methods default to `FullWindow` and provide explicit
warm-up variants. Python exposes matching `WarmupPolicy.FIRST_VALUE` and
`WarmupPolicy.FULL_WINDOW` enum values and accepts them in builder methods.

The serialized feature-vector spec version remains `1.0.0`. Compatibility with earlier
development artifacts is intentionally not preserved.

Non-window features do not accept a warm-up policy and become ready when they
first produce a value. A public feature-vector `all_ready()` aggregation is
deferred.

## Consequences

- ML-facing outputs use NaN until their configured history is available.
- Callers can explicitly retain partial-value behavior with `FirstValue`.
- Multi-window indicators can expose shorter windows before longer windows.
- Timed values and readiness reflect the extractor's global event-time
  watermark, including quiet periods and events for other symbols.
- Readiness can later be aggregated without scanning output values for NaN.
- Existing Rust constructors, serialized feature-vector specs, and Decimal builds break
  during the pre-release development phase.

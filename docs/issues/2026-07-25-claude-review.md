# fiml project review

Reviewed commit: `d3e7869` (branch `claude-review`), 2026-07-25.
Scope: whole workspace — `crates/fiml`, `crates/fiml-python`, notebooks, CI, docs.
Baseline health: `cargo test --all-features` = 104 passed, `cargo clippy --all-targets
--all-features` = clean.

The library reads as a well-organised codebase with an unusually clear separation between
the cold path (definition, validation, compilation) and the hot path (dispatch), good
module boundaries, honest doc comments, and a real parity story between Python batch and
Rust live. The findings below are ordered by how much they threaten the library's stated
purpose — producing ML features that are identical online and offline.

Every claim marked **Evidence** was reproduced against this checkout with a throwaway
binary linking the crate.

---

## 1. Feature correctness — highest priority

These affect the numbers that reach the model, so they outrank everything else.

### 1.1 Sample-window SMA divides by the full period during warm-up

`SimpleMovingAverage::update` always divides the running sum by `window.period`
(`crates/fiml/src/indicators/averages/sma/indicator.rs:158-163`); the `capacity < period`
branch is dead because `add_window` rejects that case at
`crates/fiml/src/indicators/averages/sma/indicator.rs:114`. Before the buffer holds
`period` samples the numerator is a partial sum, so the emitted value ramps linearly
towards the true average instead of being one.

**Evidence** — a 10-period SMA fed a constant price of 100.0:

```
before any event          -> [NaN]
after 1 price of 100.0    -> [10.0]
after 2 prices of 100.0   -> [20.0]
after 3 prices of 100.0   -> [30.0]
```

The first nine rows of every `sma` column are wrong by up to 10×, on a completely
different scale from the converged value. A gradient-boosted tree will happily split on
that ramp and learn "early in the stream" as a signal. The existing unit test
`moving_average_within_period` (same file, line 530) pins this behaviour as if it were
intended: it asserts `1.0` after a single update of `3.0` on a 3-period window.

The Python README states the opposite contract — "Cells are **NaN until their indicator
warms up**" (`crates/fiml-python/README.md`) — and so does the `values()` docstring at
`crates/fiml-python/src/lib.rs:559`. The extractor does pre-fill cells with NaN
(`crates/fiml/src/features/extractor.rs:53-56`), but SMA overwrites that on its very first
event.

Suggested fix: divide by `min(self.data.len(), window.period)` for a true partial mean, or
— better for ML — withhold the output until the window is full. See 1.2.

### 1.2 There is no warm-up contract, and every indicator invents its own

| indicator | value before the window is full |
|---|---|
| `Sma` | `partial_sum / period` (ramp, see 1.1) |
| `SmaTimed` | `sum / bucket_count` (true partial mean) |
| `Ema` | seeded with the first raw value |
| `Cvd` | partial cumulative sum |
| `ObvTimed`, `TradeCountTimed` | partial rolling sum |

Four different answers to the same question. For a feature store this is the decision that
most needs to be explicit, because the consumer must know which rows to drop.

Recommendation: pick one policy — "an output cell stays NaN until its window has seen
`period` samples / covered `window` milliseconds" is the one that matches the documentation
already written — implement it once in `write_outputs`
(`crates/fiml/src/features/builtin/mod.rs:54-68`) by having `value_at` return `None` until
ready, and expose the warm-up length per column to Python (e.g. `extractor.warmup_rows()`)
so training code can slice deterministically instead of guessing. `write_outputs` already
skips `None`, so most of the plumbing exists.

### 1.3 Time-based windows never decay without a matching event

`TradeCountTimed`, `ObvTimed` and `SmaTimed` only expire buckets inside `update_inner`
(`crates/fiml/src/indicators/counts/trade_count.rs:94`,
`crates/fiml/src/indicators/volume/obv/indicator.rs:228`), and their feature adapters only
call `update_inner` when a trade for their own symbol arrives
(`crates/fiml/src/features/builtin/trade_count.rs:37-42`). Nothing refreshes them on a
`Time` event or on another symbol's trade.

**Evidence** — `trade_count_timed("BTC", aggregation 1s, window 2s)` plus `day_of_week`:

```
names                = ["BTC:trade:count_timed:1000ms:2000ms", "clock:day_of_week"]
after 3 BTC trades   = [3.0, 4.0]
1 hour later (Time)  = [3.0, 4.0]
1 hour later (ETH)   = [3.0, 4.0]
```

"Trades in the last 2 seconds" still reads 3 an hour after the last trade. Live, a quiet
market keeps reporting the last burst forever; in batch, every row whose event belongs to
another symbol carries a stale as-of timestamp. Clock features refresh on every event
(`FeatureRoute::Every`), so a snapshot mixes a live clock with frozen windows — the two
disagree about what "now" is.

Recommendation: add a refresh hook (`fn observe(&mut self, now: i64, output: &mut O)`) that
runs for every dispatch on timed features — expire buckets against `now` and rewrite the
output cells — and route timed indicators into the every-event group as well as their kind
group. This is also the fix that makes multi-symbol batch extraction correct.

### 1.4 One non-finite input poisons a feature permanently

Rolling sums are maintained incrementally (`sum = sum + new - evicted`), so once NaN or ±Inf
enters `sum` it never leaves — not even after the offending sample slides out of the window.

**Evidence** — 3-period SMA:

```
price 100.0 -> [33.33]
price NaN   -> [NaN]
price 102.0 -> [NaN]
price 103.0 -> [NaN]
price 104.0 -> [NaN]   (the NaN sample has long since left the window)
```

`compute_features` validates finiteness on the Python DataFrame path
(`crates/fiml-python/python/fiml/__init__.py:209-211`), but the low-level `transform` /
`update` bindings and the entire Rust live path do not. One malformed tick from an exchange
feed disables that feature until the process restarts — and silently, because NaN columns
look like warm-up.

Recommendation: validate finiteness once at the dispatch boundary in the core
(`IndicatorFeatureVector::validate_dispatch` already exists as the natural place) and
either reject the event with a typed error or skip it under a documented policy. Belt and
braces: recompute the sum from the ring buffer when an evicted value was non-finite.

### 1.5 Incremental sums drift over long-running sessions

Same mechanism, slower failure: `SimpleMovingAverage`, `CumulativeVolumeDelta`,
`OnBalanceVolumeTimed` and `SimpleMovingAverageTimed` never recompute from their buffers,
so f64 rounding accumulates for as long as the process runs. Offline the stream is finite;
online it is not, which means the Rust and Python answers slowly diverge over a trading day
— exactly the train/serve skew the crate exists to eliminate.

Recommendation: periodically recompute the exact sum from the ring buffer (cheap: it is
bounded by the window and happens once every N updates), or use Neumaier compensated
summation in the accumulate step. Add a long-run parity test (e.g. 10M events) that asserts
the incremental value still equals a recomputed reference within a tight bound.

---

## 2. Python ↔ Rust parity — the core value proposition

### 2.1 `compute_features` dispatches every row as a trade, so price-source features are silently empty

`compute_features` hardcodes the kind column to `KIND_TRADE`
(`crates/fiml-python/python/fiml/__init__.py:252-259`). A feature set built with the
builder's default source — `FeatureSet().sma("BTCUSDT", [12])`, where `source="price"`
(`crates/fiml-python/src/lib.rs:170`) — routes to `EventKind::Price` and therefore never
updates. The column comes back all-NaN with no warning.

This is not hypothetical: the repository's own live example dispatches trades as
`Event::price` (`crates/fiml/examples/binance_trades.rs:57`) against a feature set built
with `.ema(sym, [12]).sma(sym, [12])` — the price-source default. Load that same
`FeatureSet` JSON in Python, call `compute_features`, and you get NaNs where Rust produced
values. The one failure mode the library is built to prevent is reachable by following the
repo's own examples.

Recommendation: at extractor construction (or at `compute_features` entry), compare the set
of event kinds the indicators subscribe to against the kinds the caller will actually feed,
and raise a `ValueError` naming the unreachable features. The routing information is
already available via `IndicatorSpec::route`. Optionally let `compute_features` accept a
`kind` column so mixed streams are expressible.

### 2.2 The transformer half of the pipeline has no parity story

`StandardScaler` / `ParallelTransformer` / `Pipeline` exist in Rust
(`crates/fiml/src/features/transformers/`, `crates/fiml/src/features/pipeline/`), but they
are not part of `FeatureSet`, not serialized into the parity artifact, and not exposed to
Python at all. A user who scales in Python with sklearn and serves in Rust is back to
maintaining two implementations — the exact problem the README calls out.

Also, `StandardScaler` holds a single scalar `mean`/`deviation` applied to every mapped
index (`crates/fiml/src/features/transformers/scaler.rs:3-9`), which is not what a standard
scaler is; there is no zero-deviation guard; and `ParallelTransformer` has private fields
and no constructor (`crates/fiml/src/features/transformers/mod.rs:19-29`), so it cannot be
built outside the crate. `Pipeline` is never exercised against the real extractor.

Recommendation: decide the direction now, while breaking changes are still allowed —
either make transformers first-class (per-feature parameters, serialized in the artifact,
exposed in the Python builder, sklearn-compatible `fit` on the Python side) or delete them
until there is a concrete need. Half-built public API is worse than none.

### 2.3 The JSON artifact leaks serde defaults into a cross-language contract

`notebooks/feature_set.json` shows what the parity file actually looks like:

```json
{ "symbol": "BTCUSDT",
  "indicator": { "TradeCountTimed": {
      "aggregation": { "secs": 0, "nanos": 1000000 },
      "window":      { "secs": 60, "nanos": 0 } } } }
```

Externally-tagged PascalCase variants (`"Sma"`, `"TradeCountTimed"`) sit next to
snake_case `ValueSource` values; `Duration` uses serde's `{secs, nanos}` struct even
though the Python API accepts friendly `"1ms"` strings; globals carry `"symbol": null`.

This file is the contract that pins train/serve equivalence and gets checked in next to
model weights. It should be a designed schema, not whatever `#[derive(Serialize)]` emitted:
`#[serde(tag = "type", rename_all = "snake_case")]` plus duration-as-string round-tripped
through the same parser Python uses. `FEATURE_SET_FORMAT_VERSION` machinery
(`crates/fiml/src/features/definition.rs:238-285`) is already in place and thoughtfully
done — it just deserves a schema worth versioning. Pre-1.0 is the moment.

One gap in that versioning: the version covers the *schema*, not the *semantics*. If the
warm-up rule from 1.1 changes, an old artifact silently produces different features. Worth
documenting that indicator semantics are part of the contract, and bumping on behaviour
changes.

### 2.4 `parse_tz` panics on non-ASCII input

`crates/fiml-python/src/lib.rs:100` calls `rest.split_at(1)` on a `&str` without checking
the char boundary.

**Evidence** (extracted function, verbatim logic):

```
parse_tz("Europe/Moscow") -> Err(invalid tz)        // fine
parse_tz("МСК")           -> panic: byte index 1 is not a char boundary
parse_tz("−05:00")        -> panic  // U+2212 MINUS SIGN, a normal web copy-paste
parse_tz("UTC+-5")        -> Ok(-18000000)          // "+-5" silently parsed as -5h
```

Across PyO3 a Rust panic surfaces as `pyo3_runtime.PanicException`, which does not inherit
from `ValueError` and so escapes the caller's error handling. Fix by matching on
`rest.chars().next()` and rejecting a signed body that itself starts with a sign. Worth a
sweep for other `split_at` / slicing on user-supplied strings; `parse_duration`
(`crates/fiml-python/src/lib.rs:41`) is safe because it splits on a counted run of ASCII
digits.

---

## 3. Architecture and performance

### 3.1 Dispatch cost grows with total indicator count, not with matching indicators

`run_group` walks every feature in the event's kind group
(`crates/fiml/src/features/indicator_vector.rs:103-111`) and each feature then compares its
own symbol against the event's (`ValueSource::value`,
`crates/fiml/src/features/definition.rs:49-61`). With N symbols configured, every trade
touches all N indicators of that kind to find the one that can fire.

**Evidence** — one SMA per symbol, dispatching 200k price events for a single hot symbol:

```
  1 symbol  (1 SMA each):     9.0 ns/event
 16 symbols (1 SMA each):    41.9 ns/event
 64 symbols (1 SMA each):   120.9 ns/event
```

13× more work for identical useful work. A crypto feature set covering 50–200 symbols is
routine, and this is the hot loop the whole design is organised around.

Recommendation: key the dispatch table by `(kind, symbol)` instead of `kind` alone — the
compiler already knows each indicator's symbol and route, so it can build per-symbol spans
at compile time and look up an event's span with one probe. Global/clock features keep
their every-event group. This also composes with the refresh hook from 1.3.

### 3.2 The capacity ladder buys nothing and costs a lot

`FeatureExtractor` is a macro-generated enum of eight variants, `Cap16`..`Cap128`
(`crates/fiml/src/features/extractor.rs:24-122`), each holding a
`Box<IndicatorFeatureVector<.., CAP>>` whose features live in a
`[MaybeUninit<BuiltinFeature<F>>; CAP]`.

The premise is avoiding heap allocation, but the state is boxed anyway, the output vector is
a runtime-sized `VecFeatureVector`, and every indicator's real state sits behind a
`VecDeque`. So the ladder delivers:

- Eight monomorphisations of the whole dispatch + compile path (code bloat, compile time).
- `MaybeUninit` + manual `Drop` + `assume_init_mut` in the hot loop; **37 `unsafe`
  occurrences across the crate**, most of them serving this pattern rather than the
  ring buffers.
- An arbitrary ceiling: `MAX_OUTPUTS = 128` (`crates/fiml/src/features/extractor.rs:39`).
  **Evidence**: 20 indicators × 8 windows is rejected with `output count 160 exceeds fixed
  capacity 128` — despite the outputs living in a heap `Vec`. 160 features is a small ML
  feature vector.
- Wasted, cache-hostile memory: `size_of::<BuiltinFeature<f64>>() == 624` bytes, so a
  `Cap128` extractor reserves 79,872 bytes of inline slots regardless of how many are used,
  and a 64-indicator scan touches ~40 KB — past L1, which is part of 3.1's slope.

Recommendation: replace the ladder with a single exactly-sized `Box<[BuiltinFeature<F>]>`
built during compilation. Identical allocation behaviour (one allocation, cold path, zero
per event), no `unsafe`, no ceilings, one monomorphisation, and denser scanning. This is the
largest single simplification available and it serves both "minimal memory allocation" and
"human readable code" from AGENTS.md.

Alongside it: `BuiltinFeature` is 624 bytes because the largest variant (SMA with 16 inline
window slots) sets the size for every variant, including one-cell `DayOfWeek`. Boxing the
heavy variants or moving window arrays behind a slice would shrink the scanned array
several-fold.

### 3.3 Custom features are documented but impossible

`BuiltinFeature`'s doc comment says "Users needing custom features wrap this in their own
enum (see the module docs)" (`crates/fiml/src/features/builtin/mod.rs:23-27`), and the
`Feature<F>` trait is public. But `IndicatorFeatureVector` stores
`[MaybeUninit<BuiltinFeature<F>>; M]` concretely, `compile` is `pub(crate)`, and
`from_feature_set` is the only constructor. There is no way for a downstream crate to add
an indicator without forking.

Recommendation: either make the storage generic over `Feature<F>` and expose a compiled-
indicator constructor, or remove the claim from the docs. Given the library is aimed at
quant users who will absolutely want their own signals, the former looks like the right
call — and it is much easier after 3.2.

### 3.4 Two storage strategies, one of them dead

`StackRingBuffer`, `ArrayFeatureVector`, `SimpleMovingAverage::new_stack`,
`SimpleMovingAverageTimed::new_stack`, `OnBalanceVolumeTimed::new_stack` and
`CumulativeVolumeDelta::new_stack` are fully implemented, tested and exported — and unused
by the extractor path, which is heap-only end to end. They are a maintenance surface
(including `StackRingBuffer`'s `% N` on every index, a real division for non-power-of-two
capacities) with no current consumer.

Recommendation: decide. Either wire a no-alloc embedded path that actually uses them
(and benchmark it), or drop them and keep one strategy. If they stay, use power-of-two
capacities with masking instead of `%`.

### 3.5 The Python batch path materialises the whole stream and holds the GIL

`transform` builds a `Vec<Event<f64>>` for every input row before dispatching any of them
(`crates/fiml-python/src/lib.rs:680-695`). At `size_of::<Event<f64>>() == 40` bytes, a
10M-trade backtest allocates 400 MB purely to hold events that are consumed once,
sequentially, on top of the output matrix. The all-or-nothing validation goal does not
require materialisation — a cheap first pass can check per-kind column presence and
timestamp monotonicity from the raw column slices, then a second pass can build each event
inline as it dispatches.

The whole dispatch loop also runs with the GIL held; wrapping it in `py.allow_threads` would
let other Python threads proceed during what is the longest-running call in the library.

On the pure-Python side, `compute_features` iterates every row twice in Python: once to
validate symbol strings (`crates/fiml-python/python/fiml/__init__.py:181-184`) and once to
build handles (`:242-249`). For millions of rows that dominates the runtime of an otherwise
Rust-speed operation. `pd.factorize` (or `astype("category")`) gives both the uniqueness
check and the integer codes vectorised, with one `self.symbol(name)` call per distinct
symbol.

---

## 4. Coverage gaps for the ML use case

### 4.1 The indicator set is thin, and `Float` blocks the obvious additions

Shipping today: SMA, EMA, CVD, timed SMA/OBV/trade-count, day-of-week, time-since-first-
event. Missing, in rough order of value for price prediction: returns and log-returns,
rolling standard deviation / realized volatility, z-score (needs mean + std),
high/low/range, RSI, MACD, ATR, VWAP, trade-size distribution stats, order-flow imbalance.

The blocker is structural: `Float` (`crates/fiml/src/types.rs:5-23`) exposes only
`from_usize` and `abs` plus arithmetic. No `sqrt`, `ln`, `min`, `max`, `NAN`, `is_nan`. Most
of the list above needs at least `sqrt` and `NaN` handling, and 1.1/1.2 need `NAN` too.

Recommendation: extend `Float` first (`sqrt`, `ln`, `NAN`, `is_nan`, `min`, `max`), then add
indicators. Note `rust_decimal` needs its `maths` feature for `sqrt`, so the optional
`decimal` impl (`crates/fiml/src/types.rs:45-63`) needs updating in the same change.

### 4.2 Order-book events are plumbed end to end but nothing consumes them

`EventKind::OrderBook`, `OrderBookUpdate` (`crates/fiml/src/features/event.rs:101-106`),
`KIND_ORDERBOOK`, and the `bid`/`ask` columns of `transform` all exist, and the Python
README documents them — but no `IndicatorSpec` routes to `EventKind::OrderBook`
(`crates/fiml/src/features/definition.rs:128-138`), so dispatching one is a guaranteed
no-op. Spread, mid-price, microprice and book imbalance are among the strongest short-horizon
features available, and the transport for them is already built. This looks like the highest
value-per-effort feature work in the repo.

### 4.3 Testing gaps

Unit coverage is genuinely good (104 tests, meaningful assertions, edge cases). What is
missing is the class of test that would have caught sections 1 and 2:

- Property/differential tests: rolling SMA/CVD/OBV vs a naive recompute over the same
  window, on random streams. Catches 1.1, 1.4 and 1.5 directly.
- A parity test with `output_dtype="float32"` and with a multi-symbol interleaved stream —
  the existing e2e (`crates/fiml-python/tests/test_event_replay_parity.py`) covers f64 and
  trade-only streams.
- A test that asserts warm-up cells are NaN, once 1.2 decides what that means.
- Long-run drift test (see 1.5).
- No benchmark covers the timed indicators or multi-symbol dispatch, which is where 3.1
  lives.

---

## 5. Process, docs and hygiene

- **CI installs clippy and never runs it** (`.github/workflows/ci.yml` — `components:
  rustfmt, clippy`, then only `fmt --check`, `build`, `test`). AGENTS.md requires applying
  clippy suggestions, so add `cargo clippy --all-targets --all-features -- -D warnings`.
  The tree is currently clean, so this lands green.
- CI only builds `--all-features`. A consumer using default features (no `serde`) is
  untested; add a `--no-default-features` build/test.
- **No README at the repository root.** `crates/fiml-python/README.md` is excellent and
  does a lot of heavy lifting; the Rust crate — the actual product — has neither a README
  nor crate-level docs in `lib.rs`, and there are zero doc-tests. For a library aimed at
  external users this is the biggest documentation gap.
- `docs/specs/python2027-07-14.md` — the date looks like a typo for 2026.
- Public-API typos worth fixing while breaking changes are free: `TradeSide::AgressorBuy` /
  `AgressorSell` → `Aggressor*` (`crates/fiml/src/features/event.rs:85-89`, and the Python
  constants mirror the correct spelling already), `ObvBucket::commulative_volume` →
  `cumulative_volume`, the `aggeregation` parameter name in
  `SimpleMovingAverageTimed::new_stack` / `new_heap`.
- Dead and unfinished code: the commented-out `BuiltinTransfomers` block
  (`crates/fiml/src/features/transformers/mod.rs:140-162`) should be deleted — git
  remembers it; `CumulativeVolumeDelta::update` returns a `Result` that is always `Ok`
  (`crates/fiml/src/indicators/volume/cvd/indicator.rs:97-100`); the dead `else { None }`
  branch in `SimpleMovingAverage::update`.
- `EVENT_KIND_COUNT` (`crates/fiml/src/features/event.rs:8-17`) must match the `EventKind`
  discriminants or `dispatch` indexes the group table out of bounds. The comment says so;
  a `const _: () = assert!(EventKind::Time as usize == EVENT_KIND_COUNT - 1);` would make
  the compiler say so.
- The live example aborts its stream loop on the first out-of-order timestamp
  (`crates/fiml/examples/binance_trades.rs:57`, `dispatch(...)?`). Exchange feeds do
  reorder around reconnects. There is no documented policy for late or duplicate events in
  online mode — worth deciding (reject / skip / clamp) and showing the intended handling in
  the example.
- The symbol interner is a process-global `Mutex<HashMap>` that never shrinks
  (`crates/fiml/src/symbols.rs`). Fine today (symbols are interned once, cold), but worth a
  note that `Symbol` ids are process-wide and unbounded, and that `resolve` allocates.

---

## Suggested order of work

1. **1.1 + 1.2** — fix the SMA warm-up and pick a single warm-up policy. Smallest change,
   largest effect on model quality, and it makes the documentation true.
2. **1.4** — reject non-finite input at the dispatch boundary. Cheap, prevents a silent
   production failure.
3. **2.1** — raise when a feature set's indicators are unreachable from the events being
   fed. Prevents the silent all-NaN failure the repo's own examples can produce.
4. **2.4** — the `parse_tz` panic. Two-line fix.
5. **3.2** — drop the capacity ladder for an exactly-sized boxed slice. Unblocks 3.1 and
   3.3, removes most of the `unsafe`, lifts the 128-feature ceiling.
6. **1.3** — the refresh hook for timed windows. Design work; do it after 3.2 so it lands
   in the simpler dispatch structure.
7. **3.1** — symbol-keyed dispatch.
8. **2.3** — lock the JSON schema before 1.0.
9. **4.1 + 4.2** — extend `Float`, then add volatility/returns and order-book features.
10. **5** — clippy in CI, root README, typo sweep, dead code removal (can happen in
    parallel at any point).

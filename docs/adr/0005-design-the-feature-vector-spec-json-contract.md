# ADR 0005: Design the feature-vector spec JSON contract

Status: accepted  
Date: 2026-07-29

## Context

`FeatureVectorSpec` JSON is the parity artifact shared by Python training and Rust
serving. Its original shape was produced directly by Serde derives, exposing
Rust details such as externally tagged PascalCase enum variants,
`Duration { secs, nanos }`, and `null` symbols for global indicators.

The artifact needs a stable, language-neutral schema. It must also have one
canonical output order because that order defines the model's feature-vector
columns.

## Decision

Compatibility with the old derived shape is intentionally not preserved.
Writers emit exact version `1.0`, and readers accept only that version.

The top-level shape is:

```json
{
  "version": "1.0",
  "capacity": 128,
  "length": 100,
  "checksum": "optional opaque metadata",
  "features": []
}
```

`features` contains feature groups. Every group has a required normalized
`symbol`; global features use `__global__`. Each group has a nonempty
`indicators` array. Duplicate normalized symbol groups are errors.

Each indicator uses a common strict envelope with `kind`, `source`, applicable
`warmup_policy`, optional shared `options`, and optional `outputs`. Empty
options are omitted. Scalar default outputs serialize without `outputs`; an
output `id` is written only when it differs from the deterministic default.
Empty output arrays are invalid.

Durations are integer strings with `ms`, `s`, `m`, or `h` units. Writers use
the largest exact unit. Fixed UTC offsets accept the Python builder's existing
spellings and serialize as `±HH:MM`.

Canonical order is:

1. the global feature group;
2. symbol groups by normalized symbol;
3. indicators by name and then their identity fields;
4. output cells in authored window order.

Input need not be sorted; deserialization produces a canonically ordered
runtime `FeatureVectorSpec`.

The expanded output count must equal `length`, and `capacity` must be at least
that length. Trailing capacity is
reserved model width rather than active features. `checksum` is stored and
round-tripped but is not calculated or verified. An empty feature-vector spec is valid;
an empty feature group is not.

## Module seam

Runtime definitions remain a flat, canonically ordered collection optimized for
compilation. A private `features/serde/serialization.rs` adapter owns the
hierarchical wire model.

Its interface is the existing `Serialize` and `Deserialize` implementation on
`FeatureVectorSpec`. Python delegates to that same implementation. Grouping, sorting,
strict field handling, scalar formatting, version checks, and conversion remain
private implementation details.

Serialization validates the wire contract and scope conversion. Extractor
compilation remains the single semantic-validation path for window rules,
indicator identities, capacity, and generated feature-name uniqueness.

## Consequences

- JSON no longer exposes Rust enum or `Duration` representation details.
- Rust and Python cannot develop separate serialization behavior.
- Canonical ordering is independent of builder call order and input JSON order.
- `FeatureVectorSpec` indicator storage becomes private to preserve its ordering
  invariant.
- Existing development artifacts and documentation must be regenerated.
- New top-level options require an intentional format-version decision.

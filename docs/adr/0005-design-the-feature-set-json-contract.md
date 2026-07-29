# ADR 0005: Design the feature-set JSON contract

Status: accepted  
Date: 2026-07-29

## Context

`FeatureSet` JSON is the parity artifact shared by Python training and Rust
serving. Its original shape was produced directly by Serde derives, exposing
Rust details such as externally tagged PascalCase enum variants,
`Duration { secs, nanos }`, and `null` symbols for global indicators.

The artifact needs a stable, language-neutral schema. It must also have one
canonical output order because that order defines the model's feature-vector
columns.

## Decision

Version `1.0.0` is redefined during pre-release development. Compatibility with
the old derived shape is intentionally not preserved. Writers emit `1.0.0`;
readers also accept `1.0` and compatible `1.0.x` patch versions, but reject
future minor versions until support is explicit.

The top-level shape is:

```json
{
  "version": "1.0.0",
  "features": [],
  "options": {}
}
```

`features` contains feature groups. A symbol-scoped group has a lowercase
`symbol`; a global group omits `symbol`. Each group has a nonempty `indicators`
array. There may be at most one global group and one group per normalized
symbol. Duplicate groups are errors rather than merge candidates.

Each indicator has a snake_case `name` and a required, strict `options` object.
Options mirror public builder arguments, are explicit even when they equal
defaults, and reject unknown fields. Parameterless indicators use
`"options": {}`. Output size is derived from the options and is not serialized.

Durations are integer strings with `ms`, `s`, `m`, or `h` units. Writers use
the largest exact unit. Fixed UTC offsets accept the Python builder's existing
spellings and serialize as `±HH:MM`.

Canonical order is:

1. the global feature group;
2. symbol groups by normalized symbol;
3. indicators by name and then their identity fields;
4. output cells in authored window order.

Input need not be sorted; deserialization produces a canonically ordered
runtime `FeatureSet`.

The top-level `options` object is required, empty, and strict until a real
feature-vector option exists. An empty feature set is valid; an empty feature
group is not.

## Module seam

Runtime definitions remain a flat, canonically ordered collection optimized for
compilation. A private `features/serialization` module owns the hierarchical
wire model:

```text
serialization/
├── mod.rs
├── feature_set.rs
├── feature.rs
├── indicator/
│   ├── mod.rs
│   └── options.rs
└── scalar.rs
```

Its interface is the existing `Serialize` and `Deserialize` implementation on
`FeatureSet`. Python delegates to that same implementation. Grouping, sorting,
strict field handling, scalar formatting, version checks, and conversion remain
private implementation details.

Serialization validates the wire contract and scope conversion. Extractor
compilation remains the single semantic-validation path for window rules,
indicator identities, capacity, and generated feature-name uniqueness.

## Consequences

- JSON no longer exposes Rust enum or `Duration` representation details.
- Rust and Python cannot develop separate serialization behavior.
- Canonical ordering is independent of builder call order and input JSON order.
- `FeatureSet` indicator storage becomes private to preserve its ordering
  invariant.
- Existing development artifacts and documentation must be regenerated.
- New top-level options require an intentional format-version decision.

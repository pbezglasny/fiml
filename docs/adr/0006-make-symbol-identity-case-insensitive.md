# ADR 0006: Make symbol identity case-insensitive

Status: accepted  
Date: 2026-07-29

## Context

Canonical feature groups sort by symbol and serialize symbols in lowercase. If
only configuration were normalized, an extractor configured for `btcusdt`
would not receive events interned as `BTCUSDT`.

Symbol case therefore cannot be a serialization-only concern. It is an identity
rule shared by configuration, event ingestion, dispatch, feature names, and
Python symbol handles.

## Decision

Symbol identity is ASCII case-insensitive throughout the library.
`symbols::intern` is the normalization seam: it converts ASCII uppercase
characters to lowercase before lookup or insertion. Resolved and serialized
symbols use that lowercase canonical spelling.

Rust builders and `FeatureSet::new` normalize stored symbol strings as well.
JSON duplicate-group validation runs after normalization, so groups named
`BTCUSDT` and `btcusdt` are an error. Indicator identity checks consequently
treat them as the same symbol.

Non-ASCII characters are left unchanged. Financial symbols are treated as
ASCII identifiers; locale-dependent case mapping is outside the contract.

## Consequences

- Differently cased ASCII spellings resolve to the same `Symbol`.
- Feature names use lowercase symbol segments.
- Existing tests, examples, or model artifacts containing uppercase symbols
  serialize differently.
- Normalization is paid only at cold symbol-interning and feature-definition
  paths, not during indicator updates.

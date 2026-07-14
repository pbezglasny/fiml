# ADR 0001: Make trades the high-level Python DataFrame boundary

Status: accepted  
Date: 2026-07-14

## Context

The Python API accepted both bar and trade rows through a `source` switch and
required callers to repeat every column mapping. The immediate notebook use
case has a concrete trade schema and needs one ML-ready snapshot per trade.
Supporting multiple high-level source shapes now makes validation, alignment,
and documentation wider than the demonstrated need.

## Decision

`FeatureExtractor.compute_features` accepts a pandas Trade DataFrame only. It
uses configurable column mappings with `symbol`, `ts`, `price`, and `volume`
defaults, and returns copied symbol/timestamp metadata followed by one complete
feature-vector snapshot per input trade.

File I/O remains outside `fiml`. Bars and all other event kinds remain
available through low-level `update` and `transform`.

The Python extractor imposes one global nondecreasing timestamp order across
all mutating methods. Calls validate completely before dispatch. Calculation
state stays `float64`; a pre-dispatch `output_dtype` setting chooses `float32`
or `float64` Python outputs.

## Consequences

- The common notebook call becomes `extractor.compute_features(trades)`.
- Input and output alignment have one explicit contract.
- Bar DataFrames can be added later as a separately designed boundary rather
  than another mode hidden behind `source`.
- Users remain responsible for loading, sorting, and persisting data.
- Strict pre-validation uses temporary arrays before dispatch so a bad batch
  cannot partially mutate the extractor.


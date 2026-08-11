# ADR 0007: Close the extractor runtime

Status: accepted  
Date: 2026-08-11

## Context

The Rust feature layer exposed a public `Feature<F>` update trait, a
`BuiltinFeature` enum, and concrete runtime adapter types such as `SmaFeature`.
The enum documentation suggested that downstream users could wrap the built-in
adapters in their own enum to add custom features.

That extension path was not usable with the library's extractors. Compilation
accepted only the library-owned `IndicatorSpec` variants and produced
`BuiltinFeature` values. `IndicatorFeatureVector` stored that concrete enum,
and neither compilation nor extractor construction accepted a downstream enum.
The public types therefore advertised an unsupported customization model and
made internal adapter contracts part of the API.

The library is pre-release, so removing this surface without compatibility
shims is acceptable. The fixed-capacity extractor and standalone calculation
types still serve separate allocation and composition needs and should not be
removed with the unsupported extension point.

## Decision

Extractor runtimes support the closed set of indicators represented by
library-owned `IndicatorSpec` variants. Adding an extractor indicator requires
adding it to the library's definition, compilation, serialization, builder, and
feature-vector construction paths.

The public `Feature<F>` trait is removed. The internal runtime enum is renamed
from `BuiltinFeature` to `IndicatorAdapter` and receives an inherent update
method. It and its concrete adapter types are implementation details under
`features/builtin/`; downstream crates cannot name or construct them.

`IndicatorFeatures` remains the shared interface of `FeatureExtractor` and
`IndicatorFeatureVector`, but it is sealed against downstream implementations.
The trait remains public so callers can use the common extractor methods and so
`Pipeline` can accept both library-owned extractor forms.

`IndicatorFeatureVector` remains public. It continues to provide fixed
indicator capacity and caller-chosen feature-vector storage for the closed
built-in indicator set. Standalone calculation types under `fiml::indicators`
also remain public and composable outside the extractor runtime.

No deprecated aliases, compatibility adapters, callback interface, or dynamic
trait-object extension mechanism is retained.

## Consequences

- The public API describes the customization that the extractor actually
  supports: configuring library indicators through `FeatureSet` and
  `IndicatorSpec`.
- Runtime dispatch remains static and allocation-free per event.
- Fixed-capacity and runtime-sized library extractors continue to work with
  `Pipeline`.
- Users can compose standalone calculations in their own applications but
  cannot inject them into `FeatureSet`, compilation, or library pipelines as a
  custom extractor implementation.
- Downstream code importing `Feature`, `BuiltinFeature`, or concrete adapter
  types breaks at compile time and must move to the supported configuration or
  standalone-indicator APIs.
- Feature semantics, canonical names, JSON, Python behavior, and builder output
  are unchanged.

This decision refines ADR 0002's runtime adapter boundary; it does not change
that ADR's grouped definition, compilation, or allocation contracts.

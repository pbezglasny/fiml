# ADR 0008: Serialize the pipeline spec

Status: accepted  
Date: 2026-09-02

## Context

`FeatureExtractorSpec` is the versioned configuration for raw feature extraction,
but a model consumes the ordered output of the transformations applied after
extraction. Persisting raw extraction and fitted preprocessing separately makes
the configuration vulnerable to train/serve skew.

The live `Pipeline` also contains extractor state, compiled numeric indexes,
and caller-owned vectors. Those runtime details are not a reproducible model
configuration and must not become part of a storage contract.

## Decision

`PipelineSpec` is the serialization seam. `Pipeline`, compiled scalar
operations, and the private wire representation of `TransformerDefinition`
remain non-serializable. Serde support is available only with the existing
optional `serde` feature.

The model-input artifact has its own exact version `1.0`, independent from the
nested raw feature-spec version:

```json
{
  "version": "1.0",
  "checksum": "optional model metadata",
  "feature_extractor": {
    "version": "1.0",
    "capacity": 2,
    "length": 2,
    "features": []
  },
  "model_input": {
    "capacity": 2,
    "length": 2,
    "transformations": [
      {
        "type": "identity",
        "input": "raw_day",
        "output": "day"
      },
      {
        "type": "standard_scale",
        "input": "raw_sma",
        "output": "scaled_sma",
        "mean": 4.0,
        "scale": 2.0
      }
    ]
  }
}
```

The `model_input` object owns the final vector dimensions and transformations.
Transformations use stable feature IDs rather than compiled indexes. Their
authored order is preserved because it is the final model-vector order.
Standard-scale parameters use JSON numbers deserialized as `f64`, matching the
pipeline calculation type.

The nested `FeatureExtractorSpec` is serialized through its existing canonical
adapter. Raw and model-input checksums are independent opaque strings: either
may be omitted without affecting the other, and neither is calculated or
verified by the library.

Readers strictly reject unsupported versions, unknown or missing fields,
unknown transformation types, explicit `null` optional fields, and declared
length or capacity mismatches. After structural validation, readers call
`PipelineSpec::with_metadata`; that constructor remains the single semantic
validation path for input IDs, output IDs, and fitted numeric parameters.

## Consequences

- One artifact reproducibly describes raw extraction and final model input.
- Serialization allocates only on the cold configuration path and does not
  change event-path allocation or execution.
- Runtime extractor state and caller-owned vectors cannot be restored from this
  artifact.
- `FeatureExtractorSpec` remains independently serializable.
- Model-input and raw-spec format versions may evolve independently.
- Python bindings and runtime-state persistence remain outside this decision.

## Amendment: fitted vector stages (2026-09-09, issue #93)

Writers now emit pipeline version `2.0`, requiring `model_input.stages` (possibly
empty). Strict `1.0` artifacts remain readable and must not contain stages. The
nested extractor remains at `1.0`. The original example above describes the
legacy format; [the canonical example](../example_of_store_definition.json)
uses the current format.

Scalar transformations define the base layout. Each fitted stage consumes the
complete preceding active layout. Scaler stages store effective means/scales;
PCA stages store component rows, means, output IDs, and effective whitening
divisors. Final length derives from the last stage, independently of base or
scratch width. See [the fitted-stage contract](../pipeline_transformer.md).

`PipelineSpec::with_stages` is the shared semantic validator;
`with_metadata` delegates with no stages. Python training retains sklearn
templates separately and publishes a spec only after successful fitting and
compilation. Export/load still excludes event state and training data.
The serde_json consumers maintained here enable `float_roundtrip` to retain
fitted `f64` values exactly. Other Rust consumers should enable it too.

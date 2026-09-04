# ADR 0008: Serialize the model-input spec

Status: accepted  
Date: 2026-09-02

## Context

`FeatureVectorSpec` is the versioned configuration for raw feature extraction,
but a model consumes the ordered output of the transformations applied after
extraction. Persisting raw extraction and fitted preprocessing separately makes
the configuration vulnerable to train/serve skew.

The live `Pipeline` also contains extractor state, compiled numeric indexes,
and caller-owned vectors. Those runtime details are not a reproducible model
configuration and must not become part of a storage contract.

## Decision

`ModelInputSpec` is the serialization seam. `Pipeline`, compiled scalar
operations, and the private wire representation of `TransformationDefinition`
remain non-serializable. Serde support is available only with the existing
optional `serde` feature.

The model-input artifact has its own exact version `1.0`, independent from the
nested raw feature-spec version:

```json
{
  "version": "1.0",
  "feature_vector_capacity": 2,
  "feature_vector_length": 2,
  "checksum": "optional model metadata",
  "feature_extractor": {
    "version": "1.0",
    "feature_vector_capacity": 2,
    "feature_vector_length": 2,
    "features": []
  },
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
```

Transformations use stable feature IDs rather than compiled indexes. Their
authored order is preserved because it is the final model-vector order.
Standard-scale parameters use JSON numbers deserialized as `f64`, matching the
pipeline calculation type.

The nested `FeatureVectorSpec` is serialized through its existing canonical
adapter. Raw and model-input checksums are independent opaque strings: either
may be omitted without affecting the other, and neither is calculated or
verified by the library.

Readers strictly reject unsupported versions, unknown or missing fields,
unknown transformation types, explicit `null` optional fields, and declared
length or capacity mismatches. After structural validation, readers call
`ModelInputSpec::with_metadata`; that constructor remains the single semantic
validation path for input IDs, output IDs, and fitted numeric parameters.

## Consequences

- One artifact reproducibly describes raw extraction and final model input.
- Serialization allocates only on the cold configuration path and does not
  change event-path allocation or execution.
- Runtime extractor state and caller-owned vectors cannot be restored from this
  artifact.
- `FeatureVectorSpec` remains independently serializable.
- Model-input and raw-spec format versions may evolve independently.
- Python bindings and runtime-state persistence remain outside this decision.

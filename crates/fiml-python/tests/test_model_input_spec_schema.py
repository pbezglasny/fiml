import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


DOCS = Path(__file__).parents[3] / "docs"
MODEL_SCHEMA_PATH = DOCS / "model-input-spec.schema.json"
FEATURE_SCHEMA_PATH = DOCS / "feature-vector-spec.schema.json"
MODEL_SCHEMA = json.loads(MODEL_SCHEMA_PATH.read_text())
FEATURE_SCHEMA = json.loads(FEATURE_SCHEMA_PATH.read_text())
MODEL_SCHEMA_WITH_ID = {"$id": MODEL_SCHEMA_PATH.as_uri(), **MODEL_SCHEMA}
REGISTRY = Registry().with_resource(
    FEATURE_SCHEMA_PATH.as_uri(), Resource.from_contents(FEATURE_SCHEMA)
)
VALIDATOR = Draft202012Validator(
    MODEL_SCHEMA_WITH_ID,
    registry=REGISTRY,
)


def canonical_example():
    return json.loads((DOCS / "example_of_store_definition.json").read_text())


def test_model_input_schema_is_valid_and_accepts_canonical_example():
    Draft202012Validator.check_schema(MODEL_SCHEMA)
    assert not list(VALIDATOR.iter_errors(canonical_example()))


@pytest.mark.parametrize(
    "transformation",
    [
        {"type": "identity", "input": "raw_price", "output": "price"},
        {
            "type": "standard_scale",
            "input": "raw_price",
            "output": "price",
            "mean": 10.0,
            "scale": 2.0,
        },
    ],
)
def test_model_input_schema_accepts_strict_transformation_variants(transformation):
    document = canonical_example()
    document["feature_vector_capacity"] = 1
    document["feature_vector_length"] = 1
    document["transformations"] = [transformation]
    document["raw_feature_vector_spec"]["features"][0]["indicators"][0]["outputs"][
        0
    ]["id"] = "raw_price"

    assert not list(VALIDATOR.iter_errors(document))


@pytest.mark.parametrize(
    "transformation",
    [
        {"type": "identity", "input": "raw_price"},
        {
            "type": "standard_scale",
            "input": "raw_price",
            "output": "price",
            "mean": 0.0,
            "scale": 0.0,
        },
        {"type": "unknown", "input": "raw_price", "output": "price"},
    ],
)
def test_model_input_schema_rejects_malformed_transformations(transformation):
    document = canonical_example()
    document["transformations"] = [transformation]

    assert list(VALIDATOR.iter_errors(document))

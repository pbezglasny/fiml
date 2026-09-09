import json
from pathlib import Path

import pytest
import fiml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


DOCS = Path(__file__).parents[3] / "docs"
MODEL_SCHEMA_PATH = DOCS / "pipeline-spec.schema.json"
FEATURE_SCHEMA_PATH = DOCS / "feature-extractor-spec.schema.json"
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
    spec = fiml.PipelineSpec.from_json(json.dumps(canonical_example()))
    assert json.loads(spec.to_json()) == canonical_example()


@pytest.mark.parametrize(
    "transformation",
    [
        {"type": "identity", "input": "raw_price", "output": "price"},
        {"type": "lagged", "input": "raw_price", "output": "price", "lag_window": 2},
        {"type": "lagged", "input": "raw_price", "output": "price", "lag_window": 10_000},
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
    document["model_input"]["capacity"] = 1
    document["model_input"]["length"] = 1
    document["model_input"]["transformations"] = [transformation]
    document["feature_extractor"]["features"][0]["indicators"][0]["outputs"][
        0
    ]["id"] = "raw_price"

    assert not list(VALIDATOR.iter_errors(document))


@pytest.mark.parametrize(
    "transformation",
    [
        {"type": "identity", "input": "raw_price"},
        {"type": "lagged", "input": "raw_price", "output": "price", "lag_window": 0},
        {"type": "lagged", "input": "raw_price", "output": "price", "lag_window": 10_001},
        {"type": "lagged", "input": "raw_price", "output": "price"},
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
    document["model_input"]["transformations"] = [transformation]

    assert list(VALIDATOR.iter_errors(document))


def test_schema_accepts_configured_book_replay_pipeline():
    document = json.loads((DOCS.parent / "tests/fixtures/order_book_replay.json").read_text())["pipeline"]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.PipelineSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    assert fiml.ModelInputPipeline(spec).n_features() == 5


def test_schema_accepts_fitted_stages_and_versioned_migration():
    document = json.loads((DOCS.parent / "tests/fixtures/sklearn_pipeline.json").read_text())["pipeline"]
    assert not list(VALIDATOR.iter_errors(document))
    assert json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json()) == document
    document["version"] = "1.0"
    assert list(VALIDATOR.iter_errors(document))
    with pytest.raises(ValueError, match="not allowed"):
        fiml.PipelineSpec.from_json(json.dumps(document))
    document = canonical_example()
    document["version"] = "1.0"
    del document["model_input"]["stages"]
    assert not list(VALIDATOR.iter_errors(document))
    upgraded = json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json())
    assert upgraded == canonical_example()
    document["version"] = "2.0"
    assert list(VALIDATOR.iter_errors(document))
    with pytest.raises(ValueError, match="missing field.*stages"):
        fiml.PipelineSpec.from_json(json.dumps(document))

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


def test_scalar_stages_validate_and_round_trip_in_version_2_1():
    spec = fiml.PipelineSpec.from_json(json.dumps(canonical_example()))
    input_id = spec.feature_ids()[0]
    spec.scalar_stage(fiml.ScalarStage().identity(input_id, output="selected"))
    spec.scalar_stage(fiml.ScalarStage().lagged("selected", lag_window=2, output="lag"))
    document = json.loads(spec.to_json())
    assert document["version"] == "2.1"
    assert not list(VALIDATOR.iter_errors(document))
    assert json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json()) == document
    document["version"] = "2.0"
    assert list(VALIDATOR.iter_errors(document))
    with pytest.raises(ValueError, match="require version 2.1"):
        fiml.PipelineSpec.from_json(json.dumps(document))
    document["version"] = "2.1"
    document["model_input"]["stages"][0]["transformations"][0]["extra"] = True
    assert list(VALIDATOR.iter_errors(document))
    with pytest.raises(ValueError, match="unknown field"):
        fiml.PipelineSpec.from_json(json.dumps(document))


def test_scalar_stage_references_are_validated_atomically_against_previous_outputs():
    spec = fiml.PipelineSpec.from_json(json.dumps(canonical_example()))
    first = spec.feature_ids()[0]
    spec.scalar_stage(fiml.ScalarStage().identity(first, output="selected"))
    before = spec.to_json()
    for invalid in [
        fiml.ScalarStage(),
        fiml.ScalarStage().identity(first),
        fiml.ScalarStage().identity("selected", output="new").identity("new"),
        fiml.ScalarStage().identity("selected").identity("selected"),
        fiml.ScalarStage().lagged("selected", lag_window=0),
        fiml.ScalarStage().standard_scale("selected", mean=0., scale=0.),
    ]:
        with pytest.raises(ValueError, match="stage 1"):
            spec.scalar_stage(invalid)
        assert spec.to_json() == before


def test_min_max_stage_requires_version_2_2_and_valid_state():
    document = canonical_example()
    outputs = [item["output"] for item in document["model_input"]["transformations"]]
    document["version"] = "2.2"
    document["model_input"]["stages"] = [{
        "type": "min_max_scale", "outputs": outputs,
        "scale": [1.0] * len(outputs), "min": [0.0] * len(outputs),
        "clip": [-2.0, 3.0],
    }]
    assert not list(VALIDATOR.iter_errors(document))
    assert json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json()) == document

    for version in ["2.0", "2.1"]:
        invalid = json.loads(json.dumps(document))
        invalid["version"] = version
        assert list(VALIDATOR.iter_errors(invalid))
        with pytest.raises(ValueError, match="require version 2.2"):
            fiml.PipelineSpec.from_json(json.dumps(invalid))
    for field, value in [("scale", [0.0] * len(outputs)),
                         ("min", [0.0]), ("clip", [2.0, 1.0])]:
        invalid = json.loads(json.dumps(document))
        invalid["model_input"]["stages"][0][field] = value
        with pytest.raises(ValueError, match="stage 0"):
            fiml.PipelineSpec.from_json(json.dumps(invalid))
    document["model_input"]["stages"][0]["clip"] = None
    assert list(VALIDATOR.iter_errors(document))
    with pytest.raises(ValueError):
        fiml.PipelineSpec.from_json(json.dumps(document))


def test_simple_imputer_stage_requires_version_2_3_and_valid_state():
    document = canonical_example()
    inputs = [item["output"] for item in document["model_input"]["transformations"]]
    document["version"] = "2.3"
    document["model_input"]["length"] = len(inputs) + 1
    document["model_input"]["capacity"] = len(inputs) + 1
    document["model_input"]["stages"] = [{
        "type": "simple_impute",
        "outputs": inputs + [f"missingindicator_{inputs[0]}"],
        "retained_input_indices": list(range(len(inputs))),
        "replacement_values": [0.0] * len(inputs),
        "indicator_input_indices": [0],
    }]
    assert not list(VALIDATOR.iter_errors(document))
    assert json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json()) == document

    old = json.loads(json.dumps(document))
    old["version"] = "2.2"
    assert list(VALIDATOR.iter_errors(old))
    with pytest.raises(ValueError, match="require version 2.3"):
        fiml.PipelineSpec.from_json(json.dumps(old))

    for field, value in [
        ("replacement_values", [0.0]),
        ("retained_input_indices", [1, 0] + list(range(2, len(inputs)))),
        ("indicator_input_indices", [len(inputs)]),
    ]:
        invalid = json.loads(json.dumps(document))
        invalid["model_input"]["stages"][0][field] = value
        with pytest.raises(ValueError, match="stage 0"):
            fiml.PipelineSpec.from_json(json.dumps(invalid))

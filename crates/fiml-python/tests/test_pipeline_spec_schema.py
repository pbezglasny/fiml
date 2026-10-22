"""Version 4.0 schema shapes and cross-node semantic validation."""
import copy
import json
from pathlib import Path

import fiml
import jsonschema
import pytest
from referencing import Registry, Resource

DOCS = Path(__file__).resolve().parents[3] / "docs"
SCHEMA = json.loads((DOCS / "pipeline-spec.schema.json").read_text())
RAW_SCHEMA = json.loads((DOCS / "feature-extractor-spec.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(
    SCHEMA, registry=Registry().with_resource(
        "feature-extractor-spec.schema.json", Resource.from_contents(RAW_SCHEMA)))


def document():
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="price")
    spec = fiml.PipelineSpec(raw, checksum="opaque")
    lag = spec.add(spec.root(), fiml.ScalarStage().lagged("price", lag_window=1))
    spec.output("lag", lag)
    spec.output("raw", spec.root())
    return json.loads(spec.to_json())


def test_documented_artifact_and_roundtrip():
    for data in [document(), json.loads((DOCS / "example_of_store_definition.json").read_text())]:
        VALIDATOR.validate(data)
        assert json.loads(fiml.PipelineSpec.from_json(json.dumps(data)).to_json()) == data


@pytest.mark.parametrize("field", ["version", "nodes", "outputs"])
def test_required_fields(field):
    data = document()
    del data[field]
    assert list(VALIDATOR.iter_errors(data))
    with pytest.raises(ValueError, match="missing field"):
        fiml.PipelineSpec.from_json(json.dumps(data))


@pytest.mark.parametrize("where", [(), ("nodes", 0), ("nodes", 1), ("outputs", 0), ("nodes", 1, "stage")])
def test_unknown_fields(where):
    data = document()
    target = data
    for key in where:
        target = target[key]
    target["unknown"] = True
    assert list(VALIDATOR.iter_errors(data))
    with pytest.raises(ValueError, match="unknown field"):
        fiml.PipelineSpec.from_json(json.dumps(data))


@pytest.mark.parametrize("parent", [-1, 1, 2, 999, 0.5, None, True])
def test_invalid_parents(parent):
    data = document()
    data["nodes"][1]["parent"] = parent
    with pytest.raises(ValueError):
        fiml.PipelineSpec.from_json(json.dumps(data))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(version="3.0"),
    lambda d: d.update(nodes=[]),
    lambda d: d.update(outputs=[]),
    lambda d: d.update(checksum=None),
    lambda d: d["outputs"].append(copy.deepcopy(d["outputs"][0])),
    lambda d: d["outputs"][0].update(node=999),
    lambda d: d["outputs"][0].update(name=" "),
    lambda d: d["nodes"].append(copy.deepcopy(d["nodes"][0])),
    lambda d: d["nodes"][1]["stage"]["transformations"][0].update(input="missing"),
    lambda d: d["nodes"][1]["stage"]["transformations"][0].update(lag_window=0),
])
def test_semantic_rejection(mutation):
    data = document()
    mutation(data)
    with pytest.raises(ValueError):
        fiml.PipelineSpec.from_json(json.dumps(data))

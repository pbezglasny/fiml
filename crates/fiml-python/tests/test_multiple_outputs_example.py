import json
from pathlib import Path
import runpy
import subprocess
import sys

import fiml
import numpy as np

from test_pipeline_spec_schema import VALIDATOR


def test_example_exports_a_reusable_three_output_artifact(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/multiple_outputs.py"
    artifact = tmp_path / "model-inputs.json"
    subprocess.run([sys.executable, str(example), str(artifact)], check=True)
    document = json.loads(artifact.read_text())
    VALIDATOR.validate(document)
    pipeline = fiml.ModelInputPipeline.from_json(artifact.read_text())
    assert pipeline.output_names() == ["model_a", "model_b", "decision"]
    assert pipeline.output("decision").feature_names() == ["price", "volume"]


def test_fitted_branches_match_shared_rust_fixture_and_independent_references():
    root = Path(__file__).resolve().parents[3]
    example = runpy.run_path(str(root / "crates/fiml-python/examples/multiple_outputs.py"))
    pipeline, data, fit_mask, expected = example["train"]()
    fixture = json.loads((root / "tests/fixtures/sklearn_multiple_outputs.json").read_text())
    VALIDATOR.validate(fixture["pipeline"])
    for key in ["timestamp", "price", "volume"]:
        assert fixture[key] == data[key].tolist()
    assert fixture["fit_mask"] == fit_mask.tolist()
    names = [output["name"] for output in fixture["expected"]]
    assert names == pipeline.output_names() == list(expected)
    handles = [pipeline.output(name) for name in names]

    # fit() and reload both start cold; retained output handles survive refitting/reset.
    pipeline.fit(**data, fit_mask=fit_mask)
    restored = fiml.ModelInputPipeline.from_json(json.dumps(fixture["pipeline"]))
    assert restored.symbol(fixture["symbol"]) == data["symbol"][0]
    assert restored.output_names() == names
    for current in [pipeline, restored]:
        for _ in range(2):
            for name in names:
                assert np.isnan(current.output(name).values()).all()
            actual = current.transform(**data)
            assert list(actual) == names
            for output in fixture["expected"]:
                name = output["name"]
                handle = current.output(name)
                assert handle.feature_names() == output["feature_ids"]
                assert handle.n_features() == len(output["feature_ids"])
                np.testing.assert_allclose(expected[name], output["values"], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-12)
                np.testing.assert_array_equal(handle.values(), actual[name][-1])
            if current is pipeline:
                for handle in handles:
                    np.testing.assert_array_equal(handle.values(), actual[handle.name][-1])
            current.reset()

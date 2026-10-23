import json
from pathlib import Path
import subprocess
import sys

import fiml

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

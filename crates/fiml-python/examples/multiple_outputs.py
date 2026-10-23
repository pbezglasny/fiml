"""Export one fitted artifact and read three named vectors from one replay.

Run with fiml installed (only NumPy is needed):

    python crates/fiml-python/examples/multiple_outputs.py /tmp/model-inputs.json
"""
import argparse
from pathlib import Path

import fiml
import numpy as np


def build_spec():
    raw = (fiml.FeatureExtractorSpec(capacity=4)
           .field("BTCUSDT", source="trade_price", id="price")
           .field("BTCUSDT", source="trade_volume", id="volume"))
    spec = fiml.PipelineSpec(raw)
    shared = spec.add(spec.root(), fiml.ScalarStage()
                      .sma("price", window=2, warmup=fiml.WarmupPolicy.FULL_WINDOW, output="mean"))
    # These numeric parameters stand in for an already-fitted scaler.
    model_a = spec.add(shared, fiml.ScalarStage().standard_scale("mean", mean=10., scale=2.))
    model_b = spec.add(shared, fiml.ScalarStage().lagged("mean", lag_window=1))
    spec.output("model_a", model_a)
    spec.output("model_b", model_b)
    spec.output("decision", spec.root())
    return spec


def main(artifact_path):
    artifact_path.write_text(build_spec().to_json() + "\n")
    pipeline = fiml.ModelInputPipeline.from_json(artifact_path.read_text())
    btc = pipeline.symbol("BTCUSDT")
    rows = pipeline.transform(
        kind=np.full(4, fiml.KIND_TRADE, dtype=np.uint8),
        symbol=np.full(4, btc, dtype=np.int64),
        timestamp=np.arange(4, dtype=np.int64),
        price=np.array([10., 14., 18., 22.]),
        volume=np.array([1., 2., 3., 4.]),
    )
    np.testing.assert_equal(rows["model_a"], [[np.nan], [1.], [3.], [5.]])
    np.testing.assert_equal(rows["model_b"], [[np.nan], [np.nan], [12.], [16.]])
    np.testing.assert_equal(rows["decision"], [[10., 1.], [14., 2.], [18., 3.], [22., 4.]])
    for name in pipeline.output_names():
        output = pipeline.output(name)
        np.testing.assert_equal(output.values(), rows[name][-1])
        print(f"{name} {output.feature_names()}: {output.values().tolist()}")
    print(f"Saved {artifact_path}; all three outputs came from one event replay.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="destination for the reusable JSON artifact")
    main(parser.parse_args().artifact)

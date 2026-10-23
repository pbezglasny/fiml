"""Fit two scaler/PCA branches and export one artifact with a decision output.

Run with fiml[sklearn] installed:

    python crates/fiml-python/examples/multiple_outputs.py /tmp/model-inputs.json
"""
import argparse
from pathlib import Path

import fiml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def build_spec():
    raw = (fiml.FeatureExtractorSpec(capacity=4)
           .field("BTCUSDT", source="trade_price", id="price")
           .field("BTCUSDT", source="trade_volume", id="volume"))
    spec = fiml.PipelineSpec(raw)
    indicators = fiml.ScalarStage()
    for window in [1, 3, 5]:
        indicators.sma("price", window=window, warmup=fiml.WarmupPolicy.FULL_WINDOW,
                       output=f"sma{window}")
    indicators.lagged("price", lag_window=2, output="lag2")
    shared = spec.add(spec.root(), indicators)
    imputed = spec.add(shared, SimpleImputer(strategy="median", add_indicator=True))
    scaled_a = spec.add(imputed, StandardScaler())
    model_a = spec.add(scaled_a, PCA(n_components=2, whiten=True, svd_solver="full"))
    scaled_b = spec.add(imputed, MinMaxScaler(clip=True))
    model_b = spec.add(scaled_b, PCA(n_components=1, svd_solver="full"))
    decision = spec.add(spec.root(), fiml.ScalarStage().identity("price").identity("volume"))
    spec.output("model_a", model_a)
    spec.output("model_b", model_b)
    spec.output("decision", decision)
    return spec


def train():
    """Return the fitted graph, events, fitting mask, and independent references."""
    pipeline = build_spec().build()
    prices = np.array([10., 12., 11., 15., 13., 18., 16., 20., 17., 23., 19., 25.])
    data = dict(
        kind=np.full(len(prices), fiml.KIND_TRADE, dtype=np.uint8),
        symbol=np.full(len(prices), pipeline.symbol("BTCUSDT"), dtype=np.int64),
        timestamp=np.arange(len(prices), dtype=np.int64),
        price=prices,
        volume=np.arange(1., len(prices) + 1),
    )
    # Exclude middle observations and the final holdout without dropping their history.
    fit_mask = (data["timestamp"] < 10) & (data["timestamp"] % 3 != 2)
    rows = pipeline.fit_transform(**data, fit_mask=fit_mask)

    # Compute full-window SMA and lag inputs independently, including excluded events.
    matrix = np.full((len(prices), 4), np.nan)
    for column, window in enumerate([1, 3, 5]):
        matrix[window - 1:, column] = np.convolve(prices, np.ones(window) / window, "valid")
    matrix[2:, 3] = prices[:-2]
    expected = {}
    for name, scaler, pca in [
        ("model_a", StandardScaler(), PCA(n_components=2, whiten=True, svd_solver="full")),
        ("model_b", MinMaxScaler(clip=True), PCA(n_components=1, svd_solver="full")),
    ]:
        imputer = SimpleImputer(strategy="median", add_indicator=True).fit(matrix[fit_mask])
        imputed = imputer.transform(matrix)
        scaled = scaler.fit(imputed[fit_mask]).transform(imputed)
        expected[name] = pca.fit(scaled[fit_mask]).transform(scaled)
    expected["decision"] = np.column_stack((prices, data["volume"]))
    for name in expected:
        np.testing.assert_allclose(rows[name], expected[name], rtol=1e-10, atol=1e-12)
    return pipeline, data, fit_mask, expected


def main(artifact_path):
    fitted, data, _, expected = train()
    artifact_path.write_text(fitted.to_json() + "\n")
    pipeline = fiml.ModelInputPipeline.from_json(artifact_path.read_text())
    data["symbol"] = np.full(len(data["price"]), pipeline.symbol("BTCUSDT"), dtype=np.int64)
    rows = pipeline.transform(**data)
    for name in pipeline.output_names():
        np.testing.assert_allclose(rows[name], expected[name], rtol=1e-10, atol=1e-12)
        output = pipeline.output(name)
        np.testing.assert_equal(output.values(), rows[name][-1])
        print(f"{name} {output.feature_names()}: {output.values().tolist()}")
    print(f"Saved {artifact_path}; all three outputs came from one event replay.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="destination for the reusable JSON artifact")
    main(parser.parse_args().artifact)

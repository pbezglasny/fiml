"""Fit imputation/selection/scaling/PCA and print deployable PipelineSpec JSON.

Run with fiml[sklearn] installed:
    python crates/fiml-python/examples/sklearn_pipeline.py
"""

import fiml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer


def train():
    """Return a fitted pipeline, replay events, and an independent sklearn oracle."""
    raw = fiml.FeatureExtractorSpec().field("BTCUSDT", source="trade_price", id="price")
    indicators = fiml.ScalarStage()
    for window in [1, 3, 5]:
        indicators.sma("price", window=window, output=f"sma{window}")
    indicators.lagged("price", lag_window=2, output="lag2")
    base_spec = fiml.PipelineSpec(raw)
    base_spec.output("base", base_spec.add(base_spec.root(), indicators))
    base = base_spec.build()
    spec = fiml.PipelineSpec(raw)
    node = spec.add(spec.root(), indicators)
    for estimator in [SimpleImputer(strategy="median", add_indicator=True),
                      MaxAbsScaler(clip=True), VarianceThreshold(),
                      QuantileTransformer(n_quantiles=8),
                      PCA(n_components=2, whiten=True, svd_solver="full")]:
        node = spec.add(node, estimator)
    spec.output("model", node)
    pipeline = spec.build()
    data = dict(
        kind=np.full(12, fiml.KIND_TRADE, dtype=np.uint8),
        symbol=np.full(12, pipeline.symbol("BTCUSDT"), dtype=np.int64),
        timestamp=np.arange(12, dtype=np.int64),
        price=np.array([10., 12., 11., 15., 13., 18., 16., 20., 17., 23., 19., 25.]),
        volume=np.ones(12),
    )
    base.symbol("BTCUSDT")
    matrix = base.transform(**data)["base"]
    actual = pipeline.fit_transform(**data)["model"]

    # Independently check sklearn inference on the same indicator/lag snapshots.
    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(matrix)
    scaled = MaxAbsScaler(clip=True).fit_transform(imputed)
    selected = VarianceThreshold().fit_transform(scaled)
    transformed = QuantileTransformer(n_quantiles=8).fit_transform(selected)
    pca = PCA(n_components=2, whiten=True, svd_solver="full").fit(transformed)
    expected = pca.transform(transformed)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    return pipeline, data, expected


if __name__ == "__main__":
    pipeline, _, _ = train()
    print(pipeline.to_json())

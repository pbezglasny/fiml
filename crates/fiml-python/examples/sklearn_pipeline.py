"""Fit imputation/selection/scaling/PCA and print deployable PipelineSpec JSON.

Run with fiml[sklearn] installed:
    python crates/fiml-python/examples/sklearn_pipeline.py
"""

import fiml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler


def train():
    """Return a fitted pipeline, replay events, and an independent sklearn oracle."""
    raw = fiml.FeatureExtractorSpec().sma(
        "BTCUSDT", [1, 3, 5], source="trade_price",
        warmup=fiml.WarmupPolicy.FULL_WINDOW,
    )
    spec = fiml.PipelineSpec(raw)
    for name in raw.feature_ids():
        spec.identity(name)
    spec.lagged(raw.feature_ids()[0], lag_window=2, output="lag2")
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(SimpleImputer(strategy="median", add_indicator=True), name="impute")
                .add_transformation(MaxAbsScaler(clip=True), name="scale")
                .add_transformation(VarianceThreshold(), name="variance")
                .add_transformation(PCA(n_components=2, whiten=True, svd_solver="full"), name="pca"))
    data = dict(
        kind=np.full(12, fiml.KIND_TRADE, dtype=np.uint8),
        symbol=np.full(12, pipeline.symbol("BTCUSDT"), dtype=np.int64),
        timestamp=np.arange(12, dtype=np.int64),
        price=np.array([10., 12., 11., 15., 13., 18., 16., 20., 17., 23., 19., 25.]),
        volume=np.ones(12),
    )
    base = fiml.ModelInputPipeline(spec)
    base.symbol("BTCUSDT")
    matrix = base.transform(**data)
    actual = pipeline.fit_transform(**data)

    # Independently check sklearn inference on the same indicator/lag snapshots.
    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(matrix)
    scaled = MaxAbsScaler(clip=True).fit_transform(imputed)
    selected = VarianceThreshold().fit_transform(scaled)
    pca = PCA(n_components=2, whiten=True, svd_solver="full").fit(selected)
    expected = pca.transform(selected)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    return pipeline, data, expected


if __name__ == "__main__":
    pipeline, _, _ = train()
    print(pipeline.to_json())

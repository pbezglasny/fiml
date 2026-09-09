import json
import subprocess
import sys
import runpy
from pathlib import Path

import fiml
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def base_spec(*, capacity=None, full_window=False, lag=False):
    raw = fiml.FeatureExtractorSpec(capacity=4).sma(
        "BTCUSDT", [1, 2, 4], source="trade_price",
        warmup=fiml.WarmupPolicy.FULL_WINDOW if full_window else fiml.WarmupPolicy.FIRST_VALUE,
    )
    spec = fiml.PipelineSpec(raw, capacity=capacity, checksum="training-test")
    for name in raw.feature_ids():
        spec.identity(name)
    if lag:
        spec.lagged(raw.feature_ids()[0], lag_window=2, output="lag2")
    return spec


def events(pipeline, *, offset=0.0, constant=False):
    # An unused handle before BTC catches runtime rebuilds that reorder symbols.
    assert pipeline.symbol("unused") == 0
    handle = pipeline.symbol("BTCUSDT")
    prices = np.full(24, 10.0) if constant else 10 + np.arange(24) / 3 + np.sin(np.arange(24))
    return dict(
        kind=np.full(24, fiml.KIND_TRADE, dtype=np.uint8),
        symbol=np.full(24, handle, dtype=np.int64),
        timestamp=np.arange(24, dtype=np.int64),
        price=prices + offset, volume=np.ones(24),
    )


def base_matrix(spec, data):
    base = fiml.ModelInputPipeline(spec)
    base.symbol("unused")
    base.symbol("BTCUSDT")
    return base.transform(**data)[:, :spec.active_feature_count]


@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_fit_export_and_replay_match_sklearn(whiten, dtype):
    spec = base_spec()
    scaler = StandardScaler().set_output(transform="pandas")
    pca = PCA(n_components=2, whiten=whiten, svd_solver="full")
    pipeline = (fiml.ModelInputPipeline(spec, output_dtype=dtype)
                .add_transformation(scaler, name="scale")
                .add_transformation(pca, name="pca"))
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    scaled = StandardScaler().fit(matrix).transform(matrix)
    reference = PCA(n_components=2, whiten=whiten, svd_solver="full").fit(scaled)
    expected = reference.transform(scaled)
    assert pipeline.fit(**data) is pipeline
    assert np.isnan(pipeline.values()).all()  # successful fit leaves cold state
    actual = pipeline.transform(**data)
    np.testing.assert_allclose(actual, expected.astype(dtype), rtol=1e-10 if dtype == "float64" else 1e-6, atol=1e-12 if dtype == "float64" else 1e-6)
    assert not hasattr(scaler, "scale_") and not hasattr(pca, "components_")
    assert pipeline.feature_names() == ["pca__pc0", "pca__pc1"]
    assert pipeline.active_feature_count() == pipeline.n_features() == 2
    document = json.loads(pipeline.to_json())
    assert document["version"] == "2.0"
    assert document["feature_extractor"]["version"] == "1.0"
    assert document["checksum"] == "training-test"
    assert document["model_input"]["length"] == document["model_input"]["capacity"] == 2
    assert set(document["model_input"]["stages"][1]) == {"type", "outputs", "mean", "components", "output_scale"}
    restored = fiml.ModelInputPipeline.from_json(pipeline.to_json(), output_dtype=dtype)
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)
    np.testing.assert_array_equal(pipeline.reset().transform(**data), actual)
    np.testing.assert_array_equal(pipeline.fit_transform(**data), actual)
    np.testing.assert_array_equal(pipeline.reset().compute_features(pd.DataFrame({
        "symbol": ["BTCUSDT"] * 24, "ts": data["timestamp"],
        "price": data["price"], "volume": data["volume"],
    }))[pipeline.feature_names()].to_numpy(), actual)


@pytest.mark.parametrize("with_mean,with_std", [(True, True), (True, False), (False, True), (False, False)])
def test_scaling_flags_and_constant_columns(with_mean, with_std):
    spec = base_spec(capacity=5)
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(scaler, name="scale")
    data = events(pipeline, constant=True)
    expected = scaler.fit(base_matrix(spec, data)).transform(base_matrix(spec, data))
    result = pipeline.fit_transform(**data)
    np.testing.assert_allclose(result[:, :3], expected)
    assert np.isnan(result[:, 3:]).all()
    assert pipeline.n_features() == 5


@pytest.mark.parametrize("n_components,solver", [
    (1, "full"), (0.95, "full"), (None, "full"),
    (2, "covariance_eigh"), (2, "randomized"), (2, "arpack"),
])
def test_fitted_component_count_and_pca_without_scaling(n_components, solver):
    spec = base_spec()
    options = dict(n_components=n_components, svd_solver=solver, random_state=42)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(PCA(**options), name="reduce")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    reference = PCA(**options).fit(matrix)
    actual = pipeline.fit_transform(**data)
    assert actual.shape[1] == reference.n_components_
    np.testing.assert_allclose(actual, reference.transform(matrix), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("solver", ["full", "covariance_eigh"])
def test_rank_deficient_whitening_has_bounded_roundoff_and_exact_reload(solver):
    raw = fiml.FeatureExtractorSpec().sma("BTCUSDT", [1], source="trade_price")
    raw_id = raw.feature_ids()[0]
    spec = fiml.PipelineSpec(raw)
    for name in ("a", "b", "c"):
        spec.identity(raw_id, output=name)
    options = dict(n_components=3, whiten=True, svd_solver=solver)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(PCA(**options), name="pca")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    reference = PCA(**options).fit(matrix)
    actual = pipeline.fit_transform(**data)
    expected = reference.transform(matrix)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-10, atol=1e-12)

    # Two dot products, subtraction and division in each implementation. A
    # conservative forward-error bound accounts for both reduction orders;
    # clipping the divisor to epsilon cannot make a null component well conditioned.
    eps = np.finfo(np.float64).eps
    gamma = (matrix.shape[1] + 2) * eps / (1 - (matrix.shape[1] + 2) * eps)
    magnitude = (np.abs(matrix) + np.abs(reference.mean_)) @ np.abs(reference.components_).T
    divisor = np.maximum(np.sqrt(reference.explained_variance_), eps)
    bound = 2 * gamma * magnitude / divisor
    assert np.all(np.abs(actual - expected) <= bound)

    restored = fiml.ModelInputPipeline.from_json(pipeline.to_json())
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


def test_large_offsets_and_zero_variance_whitening():
    spec = base_spec()
    for offset, constant, whiten, tolerance in [(1e12, False, False, 0.002), (0.0, True, True, 1e-12)]:
        pipeline = fiml.ModelInputPipeline(spec).add_transformation(PCA(n_components=3, whiten=whiten, svd_solver="full"), name="pca")
        data = events(pipeline, offset=offset, constant=constant)
        matrix = base_matrix(spec, data)
        reference = PCA(n_components=3, whiten=whiten, svd_solver="full").fit(matrix)
        actual = pipeline.fit_transform(**data)
        # At 1e12, projected-mean cancellation differs by a few float64 ULPs.
        np.testing.assert_allclose(actual, reference.transform(matrix), rtol=1e-10, atol=tolerance)
        assert np.isfinite(actual).all()
        if constant:
            scale = json.loads(pipeline.to_json())["model_input"]["stages"][0]["output_scale"]
            np.testing.assert_array_equal(scale, np.full(3, np.finfo(np.float64).eps))


def test_warmup_mask_lags_chunks_and_failed_refit_are_atomic():
    spec = base_spec(full_window=True, lag=True)
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(StandardScaler(), name="scale")
                .add_transformation(PCA(n_components=2), name="pca"))
    data = events(pipeline)
    with pytest.raises(ValueError, match="row 0: feature.*fit_mask"):
        pipeline.fit(**data)
    matrix = base_matrix(spec, data)
    mask = np.isfinite(matrix).all(axis=1)
    selected = StandardScaler().fit(matrix[mask]).transform(matrix[mask])
    reference = PCA(n_components=2).fit(selected).transform(selected)
    actual = pipeline.fit_transform(**data, fit_mask=mask)
    np.testing.assert_allclose(actual[mask], reference, rtol=1e-10, atol=1e-12)
    assert np.isnan(actual[~mask]).all()
    assert len(actual) == len(mask)
    before_json, before_values = pipeline.to_json(), pipeline.values()
    for invalid_mask in [np.ones(24), np.ones(23, dtype=bool), np.zeros(24, dtype=bool), np.ones((24, 1), dtype=bool)]:
        with pytest.raises(ValueError):
            pipeline.fit(**data, fit_mask=invalid_mask)
        assert pipeline.to_json() == before_json
        np.testing.assert_array_equal(pipeline.values(), before_values)
    # Scaler fitting succeeds, but PCA cannot fit two components from one row.
    one_row = np.zeros(24, dtype=bool)
    one_row[5] = True
    with pytest.raises(ValueError, match="n_components"):
        pipeline.fit(**data, fit_mask=one_row)
    assert pipeline.to_json() == before_json
    np.testing.assert_array_equal(pipeline.values(), before_values)
    bad_data = dict(data, timestamp=data["timestamp"][::-1].copy())
    with pytest.raises(ValueError, match="row 1"):
        pipeline.fit(**bad_data, fit_mask=mask)
    assert pipeline.to_json() == before_json
    with pytest.raises(ValueError, match="previous timestamp 23"):
        pipeline.update(fiml.KIND_TRADE, 1, 0, price=10., volume=1.)
    pipeline.reset()
    chunks = [pipeline.transform(**{key: value[start:end] for key, value in data.items()})
              for start, end in [(0, 9), (9, 24)]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)


def test_unfitted_guards_recipe_locking_and_cloning():
    pipeline = fiml.ModelInputPipeline(base_spec())
    estimator = PCA(n_components=2)
    pipeline.add_transformation(estimator, name="pca")
    estimator.n_components = 99  # authoring captured its own template
    data = events(pipeline)
    calls = [pipeline.values, pipeline.raw_values, pipeline.feature_names,
             pipeline.n_features, pipeline.active_feature_count, pipeline.to_spec,
             pipeline.to_json, pipeline.reset,
             lambda: pipeline.transform(**data),
             lambda: pipeline.update(fiml.KIND_TIME, 0, 0),
             lambda: pipeline.transform_order_book([]),
             lambda: pipeline.update_order_book(None),
             lambda: pipeline.compute_features(pd.DataFrame())]
    for call in calls:
        with pytest.raises(ValueError, match="not fitted"):
            call()
    assert pipeline.raw_feature_names()
    with pytest.raises(ValueError, match="duplicate stage"):
        pipeline.add_transformation(PCA(), name="pca")
    pipeline.fit(**data)
    assert pipeline.n_features() == 2
    with pytest.raises(ValueError, match="established recipe"):
        pipeline.add_transformation(StandardScaler(), name="scale")
    with pytest.raises(ValueError, match="inference-only"):
        fiml.ModelInputPipeline.from_json(pipeline.to_json()).fit(**data)
    runtime = fiml.ModelInputPipeline(base_spec())
    runtime.update(fiml.KIND_TIME, 0, 0)
    runtime.reset()
    with pytest.raises(ValueError, match="established recipe"):
        runtime.add_transformation(PCA(), name="pca")


@pytest.mark.parametrize("estimator", [object(), PCA(copy=False), StandardScaler(copy=False), type("CustomPCA", (PCA,), {})()])
def test_unsupported_estimators_do_not_change_pipeline(estimator):
    pipeline = fiml.ModelInputPipeline(base_spec())
    before = pipeline.to_json()
    with pytest.raises((TypeError, ValueError)):
        pipeline.add_transformation(estimator, name="bad")
    assert pipeline.to_json() == before


def test_artifact_inference_does_not_import_sklearn():
    pipeline = fiml.ModelInputPipeline(base_spec()).add_transformation(PCA(n_components=2), name="pca")
    pipeline.fit(**events(pipeline))
    code = """
import importlib.abc
import sys
class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise ImportError('sklearn deliberately unavailable')
sys.meta_path.insert(0, BlockSklearn())
import fiml
pipeline = fiml.ModelInputPipeline.from_json(sys.stdin.read())
handle = pipeline.symbol('BTCUSDT')
pipeline.update(fiml.KIND_TRADE, handle, 0, price=10., volume=1.)
assert pipeline.n_features() == 2
assert 'sklearn' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], input=pipeline.to_json(), text=True, check=True)


def test_shared_fixture_matches_real_sklearn_training():
    root = Path(__file__).resolve().parents[3]
    namespace = runpy.run_path(str(root / "crates/fiml-python/examples/sklearn_pipeline.py"))
    pipeline, data, expected = namespace["train"]()
    fixture = json.loads((root / "tests/fixtures/sklearn_pipeline.json").read_text())
    assert fixture["price"] == data["price"].tolist()
    assert fixture["timestamp"] == data["timestamp"].tolist()
    oracle = np.array(fixture["expected"], dtype=np.float64)
    np.testing.assert_allclose(expected, oracle, rtol=1e-10, atol=1e-12)
    restored = fiml.ModelInputPipeline.from_json(json.dumps(fixture["pipeline"]))
    restored.symbol("BTCUSDT")
    assert restored.feature_names() == pipeline.feature_names()
    np.testing.assert_allclose(restored.transform(**data), oracle, rtol=1e-10, atol=1e-12)

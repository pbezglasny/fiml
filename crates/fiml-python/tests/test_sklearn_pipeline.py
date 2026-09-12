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
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


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


def variance_spec(*, capacity=None, full_window=False):
    raw = (
        fiml.FeatureExtractorSpec()
        .sma(
            "BTCUSDT",
            [1, 4],
            source="trade_price",
            warmup=(
                fiml.WarmupPolicy.FULL_WINDOW
                if full_window
                else fiml.WarmupPolicy.FIRST_VALUE
            ),
        )
        .sma("BTCUSDT", [1], source="trade_volume")
    )
    spec = fiml.PipelineSpec(raw, capacity=capacity)
    for feature_id in raw.feature_ids():
        spec.identity(feature_id)
    return spec


def test_variance_threshold_removes_constants_preserves_layout_nans_and_json():
    spec = variance_spec(capacity=4, full_window=True)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(
        VarianceThreshold(), name="variance"
    )
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    fit_mask = np.arange(len(matrix)) >= 3
    fitted = VarianceThreshold().fit(matrix[fit_mask])
    support = fitted.get_support(indices=True)

    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)
    expected = fitted.transform(matrix)
    np.testing.assert_array_equal(actual[:, : len(support)], expected)
    assert np.isnan(actual[:, len(support) :]).all()
    assert pipeline.active_feature_count() == len(support) == 2
    assert pipeline.n_features() == 4
    assert pipeline.feature_names()[: len(support)] == np.asarray(spec.feature_ids())[
        support
    ].tolist()
    assert np.isfinite(actual[0, 0]) and np.isnan(actual[0, 1])

    document = json.loads(pipeline.to_json())
    stage = document["model_input"]["stages"][0]
    assert document["version"] == "2.5"
    assert stage == {
        "type": "select",
        "outputs": np.asarray(spec.feature_ids())[support].tolist(),
        "input_indices": support.tolist(),
    }
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


def test_variance_threshold_positive_boundary_matches_sklearn():
    spec = base_spec()
    probe = fiml.ModelInputPipeline(spec)
    data = events(probe)
    matrix = base_matrix(spec, data)
    threshold = np.min(VarianceThreshold().fit(matrix).variances_)
    fitted = VarianceThreshold(threshold=threshold).fit(matrix)
    assert 0 < fitted.get_support(indices=True).size < matrix.shape[1]

    pipeline = fiml.ModelInputPipeline(spec).add_transformation(
        VarianceThreshold(threshold=threshold), name="variance"
    )
    actual_data = events(pipeline)
    np.testing.assert_array_equal(
        pipeline.fit_transform(**actual_data), fitted.transform(matrix)
    )


@pytest.mark.parametrize("selector_first", [True, False])
def test_variance_threshold_chains_with_scaler_and_pca(selector_first):
    spec = variance_spec()
    selector = VarianceThreshold()
    scaler = StandardScaler()
    pca = PCA(n_components=1, svd_solver="full")
    pipeline = fiml.ModelInputPipeline(spec)
    stages = (selector, scaler) if selector_first else (scaler, selector)
    for index, stage in enumerate((*stages, pca)):
        pipeline.add_transformation(stage, name=f"stage_{index}")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    expected = matrix
    for stage in (*stages, pca):
        expected = stage.fit(expected).transform(expected)

    np.testing.assert_allclose(
        pipeline.fit_transform(**data), expected, rtol=1e-10, atol=1e-12
    )


def test_variance_threshold_fit_failures_are_atomic():
    pipeline = fiml.ModelInputPipeline(base_spec()).add_transformation(
        VarianceThreshold(), name="variance"
    )
    data = events(pipeline)
    pipeline.fit_transform(**data)
    before_json, before_values = pipeline.to_json(), pipeline.values()

    with pytest.raises(ValueError, match="No feature.*variance threshold"):
        pipeline.fit(**events(pipeline, constant=True))
    assert pipeline.to_json() == before_json
    np.testing.assert_array_equal(pipeline.values(), before_values)

    warmup = fiml.ModelInputPipeline(base_spec(full_window=True)).add_transformation(
        VarianceThreshold(), name="variance"
    )
    with pytest.raises(ValueError, match="must be finite for fitting"):
        warmup.fit(**events(warmup))


@pytest.mark.parametrize(
    "strategy,fill_value",
    [("mean", None), ("median", None), ("most_frequent", None), ("constant", -7.0)],
)
def test_simple_imputer_strategies_indicators_and_json_reload(strategy, fill_value):
    spec = base_spec(full_window=True)
    imputer = SimpleImputer(
        strategy=strategy, fill_value=fill_value, add_indicator=True
    )
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(imputer, name="impute")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    fitted = imputer.fit(matrix)
    expected = fitted.transform(matrix)

    actual = pipeline.fit_transform(**data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert pipeline.feature_names() == fitted.get_feature_names_out(
        spec.feature_ids()
    ).tolist()

    document = json.loads(pipeline.to_json())
    stage = document["model_input"]["stages"][0]
    assert document["version"] == "2.3"
    assert stage["type"] == "simple_impute"
    assert stage["retained_input_indices"] == [0, 1, 2]
    assert stage["indicator_input_indices"] == [1, 2]
    assert np.isfinite(stage["replacement_values"]).all()
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


@pytest.mark.parametrize("strategy", ["mean", "constant"])
@pytest.mark.parametrize("keep_empty_features", [False, True])
@pytest.mark.parametrize("add_indicator", [False, True])
def test_simple_imputer_empty_columns_match_sklearn(
    strategy, keep_empty_features, add_indicator
):
    spec = base_spec(full_window=True)
    options = dict(
        strategy=strategy,
        keep_empty_features=keep_empty_features,
        add_indicator=add_indicator,
    )
    if strategy == "constant":
        options["fill_value"] = -3.0
    imputer = SimpleImputer(**options)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(imputer, name="impute")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    fit_mask = np.arange(len(matrix)) < 2
    fitted = imputer.fit(matrix[fit_mask])

    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)
    np.testing.assert_allclose(actual, fitted.transform(matrix), rtol=1e-10, atol=1e-12)
    assert pipeline.feature_names() == fitted.get_feature_names_out(
        spec.feature_ids()
    ).tolist()


def test_simple_imputer_then_scaler_and_pca_matches_sklearn():
    spec = base_spec(full_window=True)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="full")
    pipeline = (
        fiml.ModelInputPipeline(spec)
        .add_transformation(imputer, name="impute")
        .add_transformation(scaler, name="scale")
        .add_transformation(pca, name="pca")
    )
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    imputed = imputer.fit_transform(matrix)
    scaled = scaler.fit_transform(imputed)
    expected = pca.fit(scaled).transform(scaled)

    actual = pipeline.fit_transform(**data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert np.isfinite(actual).all()


def test_simple_imputer_replaces_new_inference_nans_without_new_indicators():
    spec = base_spec(full_window=True)
    imputer = SimpleImputer(add_indicator=True)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(imputer, name="impute")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    fit_mask = np.arange(len(matrix)) >= 4
    fitted = imputer.fit(matrix[fit_mask])

    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)
    np.testing.assert_allclose(actual, fitted.transform(matrix), rtol=1e-10, atol=1e-12)
    assert np.isfinite(actual).all()
    assert not any(name.startswith("missingindicator_") for name in pipeline.feature_names())


def test_simple_imputer_rejects_infinity_and_zero_output_atomically():
    raw = fiml.FeatureExtractorSpec().sma(
        "BTCUSDT", [1], source="trade_price"
    )
    input_id = raw.feature_ids()[0]
    overflow = fiml.PipelineSpec(raw).standard_scale(
        input_id, mean=-1e308, scale=1.0, output="overflow"
    )
    pipeline = fiml.ModelInputPipeline(overflow).add_transformation(
        SimpleImputer(), name="impute"
    )
    data = events(pipeline)
    data["price"][:] = 1e308
    with pytest.raises(ValueError, match="must not be infinite"):
        pipeline.fit(**data)
    with pytest.raises(ValueError, match="not fitted"):
        pipeline.to_json()

    empty_raw = fiml.FeatureExtractorSpec().sma(
        "BTCUSDT", [4], source="trade_price",
        warmup=fiml.WarmupPolicy.FULL_WINDOW,
    )
    empty_spec = fiml.PipelineSpec(empty_raw).identity(empty_raw.feature_ids()[0])
    empty = fiml.ModelInputPipeline(empty_spec).add_transformation(
        SimpleImputer(), name="impute"
    )
    empty_data = events(empty)
    mask = np.zeros(len(empty_data["price"]), dtype=bool)
    mask[0] = True
    with pytest.raises(ValueError, match="zero output features"):
        empty.fit(**empty_data, fit_mask=mask)

    indicators = fiml.ModelInputPipeline(empty_spec).add_transformation(
        SimpleImputer(add_indicator=True), name="impute"
    )
    indicator_data = events(indicators)
    expected = SimpleImputer(add_indicator=True).fit(
        base_matrix(empty_spec, indicator_data)[mask]
    )
    actual = indicators.fit_transform(**indicator_data, fit_mask=mask)
    np.testing.assert_array_equal(
        actual, expected.transform(base_matrix(empty_spec, indicator_data))
    )
    stage = json.loads(indicators.to_json())["model_input"]["stages"][0]
    assert stage["retained_input_indices"] == []
    assert stage["replacement_values"] == []
    assert stage["indicator_input_indices"] == [0]


def test_simple_imputer_failed_downstream_refit_preserves_runtime():
    spec = base_spec(full_window=True)
    pipeline = (
        fiml.ModelInputPipeline(spec)
        .add_transformation(SimpleImputer(strategy="constant"), name="impute")
        .add_transformation(PCA(n_components=2), name="pca")
    )
    data = events(pipeline)
    pipeline.fit_transform(**data)
    before_json, before_values = pipeline.to_json(), pipeline.values()
    one_row = np.zeros(len(data["price"]), dtype=bool)
    one_row[0] = True
    with pytest.raises(ValueError, match="n_components"):
        pipeline.fit(**data, fit_mask=one_row)
    assert pipeline.to_json() == before_json
    np.testing.assert_array_equal(pipeline.values(), before_values)


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
    assert document["feature_extractor"]["version"] == "1.1"
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


@pytest.mark.parametrize(
    "with_centering,with_scaling",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_robust_scaling_flags_and_constant_columns(with_centering, with_scaling):
    spec = base_spec(capacity=5)
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(scaler, name="robust")
    data = events(pipeline, constant=True)
    matrix = base_matrix(spec, data)
    expected = scaler.fit(matrix).transform(matrix)
    result = pipeline.fit_transform(**data)
    np.testing.assert_allclose(result[:, :3], expected)
    assert np.isnan(result[:, 3:]).all()
    assert pipeline.feature_names()[:3] == spec.feature_ids()


@pytest.mark.parametrize(
    "options",
    [
        {"quantile_range": (10.0, 90.0)},
        {"quantile_range": (20.0, 80.0), "unit_variance": True},
    ],
)
def test_robust_scaler_outliers_unseen_values_and_json_reload(options):
    spec = base_spec(full_window=True)
    scaler = RobustScaler(**options)
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(scaler, name="robust")
    data = events(pipeline)
    data["price"][12] = 1_000_000.0
    data["price"][20:] = [-1_000.0, 500.0, 2_000.0, -500.0]
    matrix = base_matrix(spec, data)
    fit_mask = np.isfinite(matrix).all(axis=1) & (np.arange(len(matrix)) < 18)
    expected = scaler.fit(matrix[fit_mask]).transform(matrix)
    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(matrix))

    document = json.loads(pipeline.to_json())
    stage = document["model_input"]["stages"][0]
    np.testing.assert_array_equal(stage["mean"], scaler.center_)
    np.testing.assert_array_equal(stage["scale"], scaler.scale_)
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


@pytest.mark.parametrize("options", [{}, {"feature_range": (-3.0, -1.0)}])
@pytest.mark.parametrize("clip", [False, True])
def test_min_max_scaler_range_clipping_constants_and_nans(options, clip):
    spec = base_spec(full_window=True)
    scaler = MinMaxScaler(**options, clip=clip)
    feature_range = scaler.feature_range
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(scaler, name="minmax")
    data = events(pipeline, constant=True)
    data["price"][18:] = [5.0, 20.0, 4.0, 25.0, 3.0, 30.0]
    matrix = base_matrix(spec, data)
    fit_mask = np.isfinite(matrix).all(axis=1) & (np.arange(len(matrix)) < 18)
    fitted = scaler.fit(matrix[fit_mask])
    expected = fitted.transform(matrix)
    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    finite = actual[np.isfinite(actual)]
    if clip:
        assert finite.min() >= feature_range[0] and finite.max() <= feature_range[1]
    else:
        assert finite.min() < feature_range[0] or finite.max() > feature_range[1]

    document = json.loads(pipeline.to_json())
    stage = document["model_input"]["stages"][0]
    assert document["version"] == "2.2"
    assert stage["type"] == "min_max_scale"
    np.testing.assert_array_equal(stage["scale"], fitted.scale_)
    np.testing.assert_array_equal(stage["min"], fitted.min_)
    assert stage.get("clip") == (list(feature_range) if clip else None)
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


@pytest.mark.parametrize("clip", [False, True])
def test_max_abs_scaler_signed_constants_zeros_clipping_and_reload(clip):
    raw = (fiml.FeatureExtractorSpec()
           .sma("BTCUSDT", [4], source="trade_price", warmup=fiml.WarmupPolicy.FULL_WINDOW)
           .sma("BTCUSDT", [1], source="trade_volume"))
    spec = fiml.PipelineSpec(raw)
    price, volume = raw.feature_ids()
    spec.identity(price).identity(volume)
    scalar = (fiml.ScalarStage()
              .standard_scale(price, mean=14.0, scale=1.0, output="signed")
              .identity(volume, output="constant")
              .standard_scale(volume, mean=2.0, scale=1.0, output="zero"))
    scaler = MaxAbsScaler(clip=clip)
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(scalar, name="inputs")
                .add_transformation(scaler, name="maxabs"))
    data = events(pipeline)
    data["volume"][:] = 2.0
    data["price"][18:] = [1.0, 30.0, 2.0, 35.0, 3.0, 40.0]
    base = base_matrix(spec, data)
    matrix = np.column_stack((base[:, 0] - 14.0, base[:, 1], base[:, 1] - 2.0))
    fit_mask = np.isfinite(matrix).all(axis=1) & (np.arange(len(matrix)) < 18)
    fitted = scaler.fit(matrix[fit_mask])
    expected = fitted.transform(matrix)
    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(matrix))
    np.testing.assert_array_equal(actual[:, 2], matrix[:, 2])
    assert (actual[fit_mask, 0] < 0).any() and (actual[fit_mask, 0] > 0).any()
    unseen = actual[~fit_mask & np.isfinite(matrix).all(axis=1), 0]
    if clip:
        assert np.max(np.abs(unseen)) <= 1.0
    else:
        assert np.max(np.abs(unseen)) > 1.0
    assert pipeline.feature_names() == ["signed", "constant", "zero"]

    document = json.loads(pipeline.to_json())
    stage = document["model_input"]["stages"][1]
    assert document["version"] == ("2.2" if clip else "2.1")
    assert stage["type"] == ("min_max_scale" if clip else "standard_scale")
    if clip:
        np.testing.assert_array_equal(stage["scale"], np.reciprocal(fitted.scale_))
        np.testing.assert_array_equal(stage["min"], np.zeros(matrix.shape[1]))
        assert stage["clip"] == [-1.0, 1.0]
    else:
        np.testing.assert_array_equal(stage["mean"], np.zeros(matrix.shape[1]))
        np.testing.assert_array_equal(stage["scale"], fitted.scale_)
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


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


@pytest.mark.parametrize(
    "estimator",
    [
        object(),
        PCA(copy=False),
        StandardScaler(copy=False),
        RobustScaler(copy=False),
        MinMaxScaler(copy=False),
        MaxAbsScaler(copy=False),
        PowerTransformer(copy=False),
        SimpleImputer(copy=False),
        type("CustomPCA", (PCA,), {})(),
        type("CustomRobustScaler", (RobustScaler,), {})(),
        type("CustomMinMaxScaler", (MinMaxScaler,), {})(),
        type("CustomMaxAbsScaler", (MaxAbsScaler,), {})(),
        type("CustomPowerTransformer", (PowerTransformer,), {})(),
        type("CustomSimpleImputer", (SimpleImputer,), {})(),
        type("CustomVarianceThreshold", (VarianceThreshold,), {})(),
    ],
)
def test_unsupported_estimators_do_not_change_pipeline(estimator):
    pipeline = fiml.ModelInputPipeline(base_spec())
    before = pipeline.to_json()
    with pytest.raises((TypeError, ValueError)):
        pipeline.add_transformation(estimator, name="bad")
    assert pipeline.to_json() == before


@pytest.mark.parametrize(
    "imputer",
    [
        SimpleImputer(missing_values=0.0),
        SimpleImputer(strategy=lambda values: 0.0),
        SimpleImputer(strategy="constant", fill_value="missing"),
        SimpleImputer(strategy="constant", fill_value=np.inf),
        SimpleImputer(add_indicator=1),
        SimpleImputer(keep_empty_features=1),
    ],
)
def test_simple_imputer_rejects_unsupported_configuration(imputer):
    pipeline = fiml.ModelInputPipeline(base_spec())
    before = pipeline.to_json()
    with pytest.raises(ValueError):
        pipeline.add_transformation(imputer, name="bad")
    assert pipeline.to_json() == before


@pytest.mark.parametrize(
    "transformer",
    [PowerTransformer(method="bad"), PowerTransformer(standardize=1)],
)
def test_power_transformer_rejects_unsupported_configuration(transformer):
    pipeline = fiml.ModelInputPipeline(base_spec())
    before = pipeline.to_json()
    with pytest.raises(ValueError):
        pipeline.add_transformation(transformer, name="bad")
    assert pipeline.to_json() == before


@pytest.mark.parametrize("threshold", [-1.0, np.inf, np.nan, True, "0"])
def test_variance_threshold_rejects_invalid_threshold(threshold):
    pipeline = fiml.ModelInputPipeline(base_spec())
    before = pipeline.to_json()
    with pytest.raises(ValueError, match="finite nonnegative"):
        pipeline.add_transformation(VarianceThreshold(threshold=threshold), name="bad")
    assert pipeline.to_json() == before


@pytest.mark.parametrize("quantile_range", [(25.0, 25.0), (0.0, 75.0), (25.0, 100.0)])
def test_robust_scaler_rejects_invalid_unit_variance_range(quantile_range):
    pipeline = fiml.ModelInputPipeline(base_spec())
    scaler = RobustScaler(unit_variance=True, quantile_range=quantile_range)
    with pytest.raises(ValueError, match="requires 0 < q_min < q_max < 100"):
        pipeline.add_transformation(scaler, name="bad")


def test_robust_scaler_ignores_unit_variance_range_without_scaling():
    spec = base_spec()
    scaler = RobustScaler(
        with_scaling=False, unit_variance=True, quantile_range=(25.0, 25.0)
    )
    pipeline = fiml.ModelInputPipeline(spec).add_transformation(scaler, name="robust")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    np.testing.assert_allclose(
        pipeline.fit_transform(**data), scaler.fit(matrix).transform(matrix)
    )


@pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
@pytest.mark.parametrize("standardize", [False, True])
def test_power_transformer_matches_sklearn_and_json_reload(method, standardize):
    spec = base_spec(full_window=True)
    pipeline = fiml.ModelInputPipeline(spec)
    if method == "yeo-johnson":
        signed = fiml.ScalarStage()
        for index, feature_id in enumerate(spec.feature_ids()):
            signed.standard_scale(
                feature_id, mean=15.0, scale=1.0, output=f"signed_{index}"
            )
        pipeline.add_transformation(signed, name="signed")
    transformer = PowerTransformer(method=method, standardize=standardize)
    pipeline.add_transformation(transformer, name="power")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    if method == "yeo-johnson":
        matrix -= 15.0
    fit_mask = np.isfinite(matrix).all(axis=1)
    assert not hasattr(transformer, "lambdas_")
    fitted = PowerTransformer(method=method, standardize=standardize).fit(matrix[fit_mask])

    actual = pipeline.fit_transform(**data, fit_mask=fit_mask)
    expected = fitted.transform(matrix)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(matrix))
    document = json.loads(pipeline.to_json())
    stage = document["model_input"]["stages"][-1]
    assert document["version"] == "2.4"
    assert stage["type"] == "power_transform"
    assert stage["method"] == method
    np.testing.assert_array_equal(stage["lambdas"], fitted.lambdas_)
    if not standardize:
        assert stage["mean"] == [0.0] * matrix.shape[1]
        assert stage["scale"] == [1.0] * matrix.shape[1]
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)


def test_power_transformer_constants_and_box_cox_runtime_domain():
    spec = base_spec()
    constant = fiml.ModelInputPipeline(spec).add_transformation(
        PowerTransformer(), name="power"
    )
    constant_data = events(constant, constant=True)
    actual = constant.fit_transform(**constant_data)
    assert np.all(actual[:, :3] == 0.0)
    assert json.loads(constant.to_json())["model_input"]["stages"][0][
        "lambdas"
    ] == [1.0, 1.0, 1.0]

    raw = fiml.FeatureExtractorSpec().sma("BTCUSDT", [1], source="trade_price")
    feature_id = raw.feature_ids()[0]
    box_cox_spec = fiml.PipelineSpec(raw).identity(feature_id)
    signed = fiml.ScalarStage().standard_scale(
        feature_id, mean=15.0, scale=1.0, output="signed"
    )
    box_cox = (
        fiml.ModelInputPipeline(box_cox_spec)
        .add_transformation(signed, name="signed")
        .add_transformation(PowerTransformer(method="box-cox"), name="power")
    )
    data = events(box_cox, offset=10.0)
    data["price"][-4:] = [15.0, 14.0, 16.0, 13.0]
    matrix = data["price"][:, None] - 15.0
    fit_mask = np.arange(len(matrix)) < 18
    fitted = PowerTransformer(method="box-cox").fit(matrix[fit_mask])
    actual = box_cox.fit_transform(**data, fit_mask=fit_mask)
    valid = matrix[:, 0] > 0.0
    np.testing.assert_allclose(
        actual[valid], fitted.transform(matrix[valid]), rtol=1e-10, atol=1e-12
    )
    assert np.isnan(actual[~valid]).all()

    restored = fiml.ModelInputPipeline.from_json(box_cox.to_json())
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)
    online = fiml.ModelInputPipeline.from_json(box_cox.to_json())
    events(online)
    for kind, symbol, timestamp, price, volume in zip(
        data["kind"], data["symbol"], data["timestamp"], data["price"], data["volume"]
    ):
        online.update(kind, symbol, timestamp, price=price, volume=volume)
    np.testing.assert_array_equal(online.values()[0], actual[-1, 0])

    before_json, before_values = box_cox.to_json(), box_cox.values()
    invalid = dict(data, price=data["price"].copy())
    invalid["price"][0] = 15.0
    with pytest.raises(ValueError, match="strictly positive"):
        box_cox.fit(**invalid, fit_mask=fit_mask)
    assert box_cox.to_json() == before_json
    np.testing.assert_array_equal(box_cox.values(), before_values)


def test_artifact_inference_does_not_import_sklearn():
    pipeline = (fiml.ModelInputPipeline(base_spec())
                .add_transformation(SimpleImputer(), name="impute")
                .add_transformation(VarianceThreshold(), name="variance")
                .add_transformation(MaxAbsScaler(clip=True), name="maxabs")
                .add_transformation(PowerTransformer(), name="power")
                .add_transformation(PCA(n_components=2), name="pca")
                .add_transformation(fiml.ScalarStage().identity("pca__pc1").lagged("pca__pc0", lag_window=1), name="lags"))
    data = events(pipeline)
    pipeline.fit(**data, fit_mask=np.arange(len(data["timestamp"])) > 0)
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
import numpy as np
raw = fiml.FeatureExtractorSpec().sma('BTCUSDT', [1], source='trade_price')
base = fiml.PipelineSpec(raw).identity(raw.feature_ids()[0], output='price')
recipe = (fiml.ModelInputPipeline(base)
          .add_transformation(fiml.ScalarStage().lagged('price', lag_window=1), name='lag')
          .add_transformation(fiml.ScalarStage().standard_scale('price', mean=0., scale=2.), name='scale'))
result = recipe.fit_transform(
    np.full(2, fiml.KIND_TRADE, dtype=np.uint8),
    np.full(2, recipe.symbol('BTCUSDT'), dtype=np.int64),
    np.arange(2, dtype=np.int64), price=np.array([10., 20.]), volume=np.ones(2),
    fit_mask=np.array([False, True]),
)
np.testing.assert_equal(result, [[np.nan], [5.]])
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


def test_scalar_stages_between_estimators_preserve_excluded_event_history():
    spec = base_spec(capacity=5)
    scalars = (fiml.ScalarStage()
               .lagged("pca__pc0", lag_window=2, output="lag2")
               .identity("pca__pc1", output="now")
               .lagged("pca__pc0", lag_window=1, output="lag1"))
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(StandardScaler(), name="scale")
                .add_transformation(PCA(n_components=2, svd_solver="full"), name="pca")
                .add_transformation(scalars, name="lags")
                .add_transformation(fiml.ScalarStage()
                                    .standard_scale("lag1", mean=1., scale=2., output="scaled_lag")
                                    .identity("now")
                                    .identity("lag2"), name="select")
                .add_transformation(StandardScaler(), name="final"))
    # Pipeline captured an independent recipe, including the scalar definitions.
    scalars.identity("missing")
    data = events(pipeline)
    matrix = base_matrix(spec, data)
    mask = np.arange(len(matrix)) >= 2
    mask[::3] = False  # excluded middle events still contribute to lag history
    scaled = StandardScaler().fit(matrix[mask]).transform(matrix)
    projected = PCA(n_components=2, svd_solver="full").fit(scaled[mask]).transform(scaled)
    lag1 = np.r_[np.nan, projected[:-1, 0]]
    lag2 = np.r_[np.nan, np.nan, projected[:-2, 0]]
    selected = np.column_stack(((lag1 - 1.) / 2., projected[:, 1], lag2))
    expected = StandardScaler().fit(selected[mask]).transform(selected)
    actual = pipeline.fit_transform(**data, fit_mask=mask)
    np.testing.assert_allclose(actual[:, :3], expected, rtol=1e-10, atol=1e-12)
    assert np.isnan(actual[:, 3:]).all()
    assert pipeline.feature_names() == ["scaled_lag", "now", "lag2", "__reserved_3", "__reserved_4"]
    document = pipeline.to_json()
    assert json.loads(document)["version"] == "2.1"
    restored = fiml.ModelInputPipeline.from_json(document)
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)
    pipeline.reset()
    chunks = [pipeline.transform(**{key: value[start:end] for key, value in data.items()})
              for start, end in [(0, 1), (1, 7), (7, 24)]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)
    before = pipeline.values()
    with pytest.raises(ValueError, match="must be finite"):
        pipeline.fit(**data)  # downstream lag warm-up requires an explicit mask
    assert pipeline.to_json() == document
    np.testing.assert_array_equal(pipeline.values(), before)
    np.testing.assert_array_equal(pipeline.fit_transform(**data, fit_mask=mask), actual)


def test_scalar_selection_can_remove_warmup_before_fitting():
    spec = base_spec(lag=True)
    first = spec.feature_ids()[0]
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(fiml.ScalarStage().identity(first, output="selected"), name="select")
                .add_transformation(StandardScaler(), name="scale"))
    data = events(pipeline)
    expected = StandardScaler().fit_transform(base_matrix(spec, data)[:, :1])
    np.testing.assert_allclose(pipeline.fit_transform(**data), expected, atol=1e-12)


def test_training_prefix_can_be_wider_than_explicit_final_capacity():
    raw = fiml.FeatureExtractorSpec().sma("BTCUSDT", [1], source="trade_price")
    spec = fiml.PipelineSpec(raw, capacity=1).identity(raw.feature_ids()[0], output="price")
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(fiml.ScalarStage()
                                    .identity("price")
                                    .lagged("price", lag_window=1, output="lag"), name="expand")
                .add_transformation(PCA(n_components=1, svd_solver="full"), name="pca"))
    data = events(pipeline)
    mask = np.arange(len(data["price"])) > 0
    matrix = np.column_stack((data["price"], np.r_[np.nan, data["price"][:-1]]))
    reference = PCA(n_components=1, svd_solver="full").fit_transform(matrix[mask])
    actual = pipeline.fit_transform(**data, fit_mask=mask)
    assert actual.shape == (len(mask), 1)
    np.testing.assert_allclose(actual[mask], reference, rtol=1e-10, atol=1e-12)
    assert np.isnan(actual[0]).all()
    assert pipeline.to_spec().capacity == 1
    restored = fiml.ModelInputPipeline.from_json(pipeline.to_json())
    events(restored)
    np.testing.assert_array_equal(restored.transform(**data), actual)
    too_wide = (fiml.ModelInputPipeline(spec)
                .add_transformation(fiml.ScalarStage().identity("price").identity("price", output="copy"), name="expand"))
    with pytest.raises(ValueError, match="capacity"):
        too_wide.fit(**events(too_wide))
    with pytest.raises(ValueError, match="not fitted"):
        too_wide.to_json()

"""Public branching graph contract, ownership, training and lifecycle checks."""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import fiml


def raw_spec():
    return (fiml.FeatureExtractorSpec(capacity=4)
            .field("BTC", source="trade_price", id="price")
            .field("BTC", source="trade_volume", id="volume"))


def events(pipeline, size=16):
    return dict(kind=np.full(size, fiml.KIND_TRADE, dtype=np.uint8),
                symbol=np.full(size, pipeline.symbol("BTC"), dtype=np.int64),
                timestamp=np.arange(size, dtype=np.int64),
                price=np.arange(size, dtype=float) + 5 + np.sin(np.arange(size)),
                volume=np.arange(size, dtype=float) ** 2 + 2)


def graph(*, learned=False):
    spec = fiml.PipelineSpec(raw_spec())
    root = spec.root()
    shared = spec.add(root, PCA(n_components=2)) if learned else spec.add(
        root, fiml.ScalarStage().lagged("price", lag_window=1, output="price"))
    selected = spec.add(shared, fiml.ScalarStage().identity("pc0" if learned else "price"))
    spec.output("selected", selected)
    spec.output("shared", shared)
    spec.output("alias", shared)
    spec.output("raw", root)
    return spec


def test_all_outputs_update_once_and_batch_is_aligned():
    pipeline = graph().build()
    data = events(pipeline)
    result = pipeline.transform(**data)
    assert list(result) == ["selected", "shared", "alias", "raw"]
    expected = np.r_[np.nan, data["price"][:-1]][:, None]
    for name in ["selected", "shared", "alias"]:
        np.testing.assert_equal(result[name], expected)
        np.testing.assert_equal(pipeline.output(name).values(), expected[-1])
    np.testing.assert_equal(result["raw"], np.column_stack((data["price"], data["volume"])))
    assert not np.shares_memory(result["shared"], result["alias"])
    assert pipeline.raw_values().shape == (4,)
    assert not hasattr(pipeline, "values")
    with pytest.raises(ValueError, match="unknown output"):
        pipeline.output("missing")


def test_single_output_empty_batch_and_dtype():
    spec = fiml.PipelineSpec(raw_spec())
    spec.output("raw", spec.root())
    pipeline = spec.build(np.float32)
    result = pipeline.transform(**events(pipeline, 0))
    assert list(result) == ["raw"]
    assert result["raw"].shape == (0, 2)
    assert result["raw"].dtype == np.float32
    pipeline.output_dtype = "float64"
    pipeline.transform(**events(pipeline))
    with pytest.raises(ValueError, match="cannot be changed"):
        pipeline.output_dtype = "float32"


def test_builder_snapshots_inputs_builds_and_rejects_foreign_handles():
    raw = raw_spec()
    spec = fiml.PipelineSpec(raw)
    scalar = fiml.ScalarStage().identity("price")
    node = spec.add(spec.root(), scalar)
    spec.output("price", node)
    scalar.identity("volume")
    raw.day_of_week()
    first, second = spec.build(), spec.build()
    spec.output("raw", spec.root())
    assert first.output_names() == second.output_names() == ["price"]
    first.transform(**events(first))
    assert np.isnan(second.output("price").values()).all()
    assert first.output("price").feature_names() == ["price"]
    other = fiml.PipelineSpec(raw_spec())
    with pytest.raises(ValueError, match="another pipeline builder"):
        other.add(node, fiml.ScalarStage().identity("price"))
    with pytest.raises(ValueError, match="another pipeline builder"):
        other.output("foreign", node)
    with pytest.raises(ValueError, match="duplicate"):
        spec.output("price", spec.root())


@pytest.mark.parametrize("name", ["", "   ", "__reserved_0"])
def test_invalid_output_names(name):
    spec = fiml.PipelineSpec(raw_spec())
    with pytest.raises(ValueError, match="output name"):
        spec.output(name, spec.root())


@pytest.mark.parametrize("method", ["build", "to_json"])
def test_no_outputs_and_empty_root_are_rejected(method):
    with pytest.raises(ValueError, match="at least one named output"):
        getattr(fiml.PipelineSpec(raw_spec()), method)()
    spec = fiml.PipelineSpec(fiml.FeatureExtractorSpec())
    spec.output("empty", spec.root())
    with pytest.raises(ValueError, match=r"output ['\"]empty['\"].*active feature"):
        getattr(spec, method)()


def test_numeric_spec_export_does_not_construct_runtime(monkeypatch):
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("exporting a specification must not allocate runtime histories")

    monkeypatch.setattr(fiml, "_ModelInputPipeline", unexpected_runtime)
    spec = fiml.PipelineSpec(raw_spec(), checksum="numeric-export")
    for index in range(4):
        node = spec.add(spec.root(), fiml.ScalarStage().rolling_std_timed(
            "price", window="999998ms"))
        spec.output(f"std{index}", node)
    spec.output("alias", node)
    spec.output("raw", spec.root())

    artifact = spec.to_json()
    document = json.loads(artifact)
    assert document["version"] == "4.0"
    assert document["checksum"] == "numeric-export"
    assert len(document["nodes"]) == 5
    assert [node["parent"] for node in document["nodes"][1:]] == [0] * 4
    assert document["outputs"] == [
        *[{"name": f"std{index}", "node": index + 1} for index in range(4)],
        {"name": "alias", "node": 4},
        {"name": "raw", "node": 0},
    ]
    assert spec.to_json() == artifact
    assert fiml.PipelineSpec.from_json(artifact).to_json() == artifact


def test_unfitted_spec_cannot_be_exported():
    with pytest.raises(ValueError, match="not fitted"):
        graph(learned=True).to_json()


def test_fit_shares_estimator_and_keeps_output_handles(monkeypatch):
    calls = []
    original = PCA.fit
    def fit(self, matrix, *args, **kwargs):
        calls.append(matrix.copy())
        return original(self, matrix, *args, **kwargs)
    monkeypatch.setattr(PCA, "fit", fit)
    pipeline = graph(learned=True).build()
    handle = pipeline.output("shared")
    data = events(pipeline)
    with pytest.raises(ValueError, match="not fitted"):
        handle.values()
    mask = np.arange(16) >= 3
    result = pipeline.fit_transform(**data, fit_mask=mask)
    assert len(calls) == 1
    expected_input = np.column_stack((data["price"], data["volume"]))
    np.testing.assert_equal(calls[0], expected_input[mask])
    expected = original(PCA(n_components=2), expected_input[mask]).transform(expected_input)
    np.testing.assert_allclose(result["shared"], expected, atol=1e-12)
    np.testing.assert_equal(result["selected"], result["shared"][:, :1])
    assert handle.feature_names() == ["pc0", "pc1"]
    snapshot = handle.values()
    pipeline.reset()
    assert np.isnan(handle.values()).all()
    np.testing.assert_equal(snapshot, result["shared"][-1])
    pipeline.fit(**data, fit_mask=mask)
    assert np.isnan(handle.values()).all()
    pipeline.transform(**data)
    before, values = pipeline.to_json(), handle.values()
    bad = dict(data, price=np.full(16, np.inf))
    with pytest.raises(ValueError):
        pipeline.fit(**bad, fit_mask=mask)
    assert pipeline.to_json() == before
    np.testing.assert_equal(handle.values(), values)


def test_independent_pca_nodes_reuse_local_names_and_snapshot_estimators(monkeypatch):
    spec = fiml.PipelineSpec(raw_spec())
    estimator = PCA(n_components=1)
    for name in ["a", "b"]:
        spec.output(name, spec.add(spec.root(), estimator))
    estimator.n_components = 2
    pipeline = spec.build()
    calls = []
    original = PCA.fit
    def fit(self, matrix, *args, **kwargs):
        calls.append(self)
        return original(self, matrix, *args, **kwargs)
    monkeypatch.setattr(PCA, "fit", fit)
    pipeline.fit(**events(pipeline))
    assert len(calls) == 2 and calls[0] is not calls[1]
    assert pipeline.output("a").feature_names() == pipeline.output("b").feature_names() == ["pc0"]
    assert not hasattr(estimator, "components_")


def test_history_replays_excluded_rows_and_multiple_learning_passes():
    spec = fiml.PipelineSpec(raw_spec())
    lag = spec.add(spec.root(), fiml.ScalarStage().lagged("price", lag_window=2, output="lag"))
    scaled = spec.add(lag, StandardScaler())
    projected = spec.add(scaled, PCA(n_components=1))
    spec.output("model", projected)
    spec.output("lag", lag)
    pipeline = spec.build()
    data = events(pipeline)
    mask = np.arange(16) % 2 == 0
    mask[:4] = False
    result = pipeline.fit_transform(**data, fit_mask=mask)
    lagged = np.r_[np.nan, np.nan, data["price"][:-2]][:, None]
    np.testing.assert_equal(result["lag"], lagged)
    scaler = StandardScaler().fit(lagged[mask])
    pca = PCA(n_components=1).fit(scaler.transform(lagged[mask]))
    np.testing.assert_allclose(result["model"][2:], pca.transform(scaler.transform(lagged[2:])), atol=1e-12)


def test_failed_refit_missing_component_preserves_layout_and_state():
    spec = fiml.PipelineSpec(raw_spec())
    pca = spec.add(spec.root(), PCA(n_components=0.95))
    spec.output("first", pca)
    spec.output("second", spec.add(pca, fiml.ScalarStage().identity("pc1")))
    pipeline = spec.build()
    data = events(pipeline)
    data["price"] = np.tile([-1., -1., 1., 1.], 4)
    data["volume"] = np.tile([-1., 1., -1., 1.], 4)
    pipeline.fit_transform(**data)
    output = pipeline.output("second")
    saved, values = pipeline.to_json(), output.values()
    data["volume"] = data["price"].copy()
    with pytest.raises(ValueError, match="pc1|input feature ID"):
        pipeline.fit(**data)
    assert pipeline.to_json() == saved
    np.testing.assert_equal(output.values(), values)


def test_json_cold_roundtrip_order_and_dataframe_results():
    pipeline = graph(learned=True).build()
    data = events(pipeline)
    result = pipeline.fit_transform(**data)
    document = json.loads(pipeline.to_json())
    assert document["version"] == "4.0"
    assert document["nodes"][1]["stage"]["outputs"] == ["pc0", "pc1"]
    assert [out["name"] for out in document["outputs"]] == list(result)
    restored = fiml.ModelInputPipeline.from_json(pipeline.to_json())
    assert restored.to_json() == pipeline.to_json()
    assert np.isnan(restored.output("raw").values()).all()
    restored.symbol("BTC")
    for name, values in restored.transform(**data).items():
        np.testing.assert_equal(values, result[name])
    restored.reset()
    frame = pd.DataFrame(dict(symbol=["BTC"]*16, ts=data["timestamp"],
                              price=data["price"], volume=data["volume"]))
    frames = restored.compute_features(frame)
    assert list(frames) == list(result)
    for name, values in result.items():
        np.testing.assert_equal(frames[name].iloc[:, 2:].to_numpy(), values)


def test_symbol_aliases_keep_handles_before_fit_and_after_reset():
    pipeline = graph(learned=True).build()
    assert pipeline.symbol("UNUSED") == pipeline.symbol("unused") == 0
    assert pipeline.symbol("BTC") == pipeline.symbol("btc") == 1
    data = events(pipeline)
    pipeline.fit_transform(**data)
    assert pipeline.symbol("BTC") == pipeline.symbol("btc") == 1
    pipeline.reset()
    pipeline.transform(**data)
    assert pipeline.symbol("UNUSED") == 0


def test_refit_updates_existing_output_metadata_without_affecting_other_builds():
    spec = fiml.PipelineSpec(raw_spec())
    projected = spec.add(spec.root(), PCA(n_components=0.95))
    spec.output("model", projected)
    first, second = spec.build(), spec.build()
    handle = first.output("model")
    data = events(first)
    data["price"] = np.tile([-1., -1., 1., 1.], 4)
    data["volume"] = np.tile([-1., 1., -1., 1.], 4)
    first.fit_transform(**data)
    snapshot = handle.values()
    assert handle.feature_names() == ["pc0", "pc1"]
    data["volume"] = data["price"].copy()
    first.fit(**data)
    assert handle.feature_names() == ["pc0"]
    assert handle.values().shape == (1,) and np.isnan(handle.values()).all()
    assert snapshot.shape == (2,) and np.isfinite(snapshot).all()
    with pytest.raises(ValueError, match="not fitted"):
        second.output("model").values()

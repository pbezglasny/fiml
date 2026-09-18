import json

import fiml
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from test_pipeline_spec_schema import VALIDATOR


@pytest.mark.parametrize("method", ["sma", "ema"])
@pytest.mark.parametrize("warmup", [fiml.WarmupPolicy.FULL_WINDOW, fiml.WarmupPolicy.FIRST_VALUE])
def test_observation_sampling_streaming_chunks_reset_and_fit(method, warmup):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="price")
    spec = fiml.PipelineSpec(raw).standard_scale("price", mean=10.0, scale=2.0)
    stage = getattr(fiml.ScalarStage(), method)("price", window=2, warmup=warmup)
    spec.scalar_stage(stage)
    runtime = fiml.ModelInputPipeline(spec)
    btc, eth = runtime.symbol("BTC"), runtime.symbol("ETH")
    data = dict(kind=np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.uint8),
                symbol=np.array([btc, eth, btc, eth, btc, btc, btc], dtype=np.int64),
                timestamp=np.arange(7, dtype=np.int64),
                price=np.array([10., 99., 10., 99., 14., 18., 22.]), volume=np.ones(7))
    data["kind"][:] = fiml.KIND_TRADE
    actual = runtime.transform(**data)
    expected = ([np.nan, np.nan, 0., 0., 1., 3., 5.] if method == "sma"
                else [np.nan, np.nan, 0., 0., 4/3, 28/9, 136/27])
    if warmup == fiml.WarmupPolicy.FIRST_VALUE:
        expected[:2] = [0., 0.]
    np.testing.assert_allclose(actual[:, 0], expected, equal_nan=True)
    runtime.reset()
    chunks = [runtime.transform(**{k: v[a:b] for k, v in data.items()}) for a, b in [(0,1), (1,4), (4,7)]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)
    runtime.reset()
    rows = []
    for i in range(7):
        runtime.update(int(data["kind"][i]), int(data["symbol"][i]), i, price=float(data["price"][i]), volume=1.)
        rows.append(runtime.values())
    np.testing.assert_array_equal(rows, actual)
    document = json.loads(runtime.to_json())
    assert document["version"] == "3.0" and document["feature_extractor"]["version"] == "2.0"
    assert not list(VALIDATOR.iter_errors(document))
    restored = fiml.ModelInputPipeline.from_json(json.dumps(document))
    restored.symbol("BTC"); restored.symbol("ETH")
    assert np.isnan(restored.values()).all()
    np.testing.assert_array_equal(restored.transform(**data), actual)
    # Excluded rows still contribute to the moving average's history.
    mask = np.array([False, False, True, False, True, False, True])
    fitting_base = fiml.PipelineSpec(raw).standard_scale("price", mean=10.0, scale=2.0)
    fitted = (fiml.ModelInputPipeline(fitting_base).add_transformation(stage, name="average")
              .add_transformation(StandardScaler(), name="scale"))
    fitted.symbol("BTC"); fitted.symbol("ETH")
    result = fitted.fit_transform(**data, fit_mask=mask)
    oracle = StandardScaler().fit(actual[mask]).transform(actual)
    np.testing.assert_allclose(result, oracle, equal_nan=True)


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_obsolete_extractors_have_migration_guidance(version):
    raw = fiml.FeatureExtractorSpec().field("BTC")
    document = json.loads(raw.to_json())
    document["version"] = version
    with pytest.raises(ValueError, match="migrate.*field extraction"):
        fiml.FeatureExtractorSpec.from_json(json.dumps(document))


@pytest.mark.parametrize("method", ["sma", "ema"])
def test_average_schema_validation_and_no_lag_cap(method):
    raw = fiml.FeatureExtractorSpec().field("BTC", id="price")
    spec = getattr(fiml.PipelineSpec(raw), method)("price", window=10_001)
    document = json.loads(spec.to_json())
    assert not list(VALIDATOR.iter_errors(document))
    for invalid in [0, -1, 1.5, None]:
        document["model_input"]["transformations"][0]["window"] = invalid
        assert list(VALIDATOR.iter_errors(document))
        with pytest.raises(ValueError):
            fiml.PipelineSpec.from_json(json.dumps(document))

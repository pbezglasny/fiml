import subprocess
import sys

import fiml
import numpy as np
import pytest


def test_symbol_capacity_is_recoverable():
    # Exhaust the process-wide interner without affecting the rest of pytest.
    result = subprocess.run(
        [sys.executable, __file__], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def check_capacity():
    raw = fiml.FeatureVectorSpec().sma("capacity-0", [1])
    model = fiml.ModelInputSpec(raw).identity(raw.feature_ids()[0])
    extractor = fiml.FeatureExtractor(raw)
    pipeline = fiml.ModelInputPipeline(model)
    raw_json = raw.to_json()
    model_json = model.to_json()
    for runtime in [extractor, pipeline]:
        runtime.update(fiml.KIND_PRICE, runtime.symbol("capacity-0"), 0, price=100.0)
    for index in range(1, 511):
        assert extractor.symbol(f"capacity-{index}") == index

    error = "symbol count 513 exceeds limit 512"
    for runtime in [extractor, pipeline]:
        for name in ["overflow", "overflow", "another-overflow"]:
            with pytest.raises(ValueError, match=error):
                runtime.symbol(name)
        np.testing.assert_equal(runtime.values(), [100.0])
        assert runtime.symbol("CAPACITY-0") == 0
        runtime.update(fiml.KIND_PRICE, runtime.symbol("capacity-0"), 1, price=101.0)
        np.testing.assert_equal(runtime.values(), [101.0])
    np.testing.assert_equal(pipeline.raw_values(), [101.0])
    assert extractor.symbol("__GLOBAL__") == 511
    assert pipeline.symbol("__global__") == 1

    for method, kwargs in [
        ("sma", {"windows": [1]}),
        ("ema", {"windows": [1]}),
        ("cvd", {"windows": [1]}),
        ("sma_timed", {"aggregation": "1s", "windows": ["2s"]}),
        ("obv_timed", {"aggregation": "1s", "windows": ["2s"]}),
        ("trade_count_timed", {"aggregation": "1s", "window": "2s"}),
    ]:
        spec = fiml.FeatureVectorSpec()
        before = spec.to_json()
        with pytest.raises(ValueError, match=error):
            getattr(spec, method)("overflow", **kwargs)
        assert spec.to_json() == before
        getattr(spec, method)("CAPACITY-0", **kwargs)
        assert spec.output_count() == 1

    for cls, text in [
        (fiml.FeatureVectorSpec, raw_json),
        (fiml.FeatureExtractor, raw_json),
        (fiml.ModelInputSpec, model_json),
        (fiml.ModelInputPipeline, model_json),
    ]:
        with pytest.raises(ValueError, match=error):
            cls.from_json(text.replace("capacity-0", "overflow"))
        cls.from_json(text)
    assert raw.to_json() == raw_json
    assert model.to_json() == model_json
    for index in range(511):
        assert extractor.symbol(f"CAPACITY-{index}") == index


if __name__ == "__main__":
    check_capacity()

import json

import numpy as np
import pandas as pd
import pytest

import fiml


def configured_spec(capacity=4):
    return fiml.FeatureExtractorSpec(capacity=capacity, checksum="opaque").sma(
        "BTCUSDT",
        [2, 3],
        source="trade_price",
        warmup=fiml.WarmupPolicy.FIRST_VALUE,
    )


def test_fluent_spec_uses_core_json_and_round_trips_metadata():
    feature_extractor_spec = configured_spec()

    assert feature_extractor_spec.capacity == 4
    assert feature_extractor_spec.active_feature_count == 2
    assert feature_extractor_spec.output_count() == 2
    assert feature_extractor_spec.checksum == "opaque"

    document = json.loads(feature_extractor_spec.to_json())
    assert document["version"] == "1.1"
    assert document["capacity"] == 4
    assert document["length"] == 2
    assert document["checksum"] == "opaque"
    assert document["required_events"] == [{"symbol": "btcusdt", "event": "trade"}]
    assert [output["window"] for output in document["features"][0]["indicators"][0]["outputs"]] == [2, 3]
    assert all("id" not in output for output in document["features"][0]["indicators"][0]["outputs"])

    restored = fiml.FeatureExtractorSpec.from_json(feature_extractor_spec.to_json())
    assert json.loads(restored.to_json()) == document


def test_feature_ids_returns_only_active_stable_ids():
    feature_extractor_spec = configured_spec()

    assert feature_extractor_spec.feature_ids() == fiml.FeatureExtractor(
        feature_extractor_spec
    ).feature_names()[:2]


def test_omitted_capacity_tracks_outputs_and_explicit_capacity_is_fixed():
    dynamic = fiml.FeatureExtractorSpec().sma("BTCUSDT", [2]).ema("BTCUSDT", [3, 4])
    assert dynamic.capacity == 3
    assert dynamic.active_feature_count == 3

    fixed = fiml.FeatureExtractorSpec(capacity=1).day_of_week()
    with pytest.raises(ValueError, match="capacity 1 is smaller"):
        fixed.time_since_first_event_of_day()


def test_reserved_cells_are_named_and_remain_nan_in_arrays_and_dataframes():
    extractor = fiml.FeatureExtractor(configured_spec())
    names = extractor.feature_names()
    assert extractor.n_features() == 4
    assert extractor.active_feature_count() == 2
    assert names[-2:] == ["__reserved_2", "__reserved_3"]
    assert np.isnan(extractor.values()[2:]).all()

    btc = extractor.symbol("BTCUSDT")
    matrix = extractor.transform(
        np.array([fiml.KIND_TRADE], dtype=np.uint8),
        np.array([btc], dtype=np.int64),
        np.array([1], dtype=np.int64),
        price=np.array([10.0]),
        volume=np.array([1.0]),
    )
    assert matrix.shape == (1, 4)
    assert np.isnan(matrix[:, 2:]).all()

    frame_extractor = fiml.FeatureExtractor(configured_spec())
    frame = frame_extractor.compute_features(
        pd.DataFrame({"symbol": ["BTCUSDT"], "ts": [1], "price": [10.0], "volume": [1.0]})
    )
    assert list(frame.columns[-2:]) == ["__reserved_2", "__reserved_3"]
    assert frame[["__reserved_2", "__reserved_3"]].isna().all().all()


def test_custom_ids_and_direct_extractor_loading():
    document = json.loads(configured_spec(capacity=2).to_json())
    document["features"][0]["indicators"][0]["outputs"][0]["id"] = "model_price"
    text = json.dumps(document)

    extractor = fiml.FeatureExtractor.from_json(text, output_dtype="float32")
    assert extractor.feature_names()[0] == "model_price"
    assert extractor.output_dtype == "float32"


def test_reserved_id_namespace_is_rejected_from_json():
    document = json.loads(configured_spec(capacity=2).to_json())
    document["features"][0]["indicators"][0]["outputs"][0]["id"] = "__reserved_0"
    with pytest.raises(ValueError, match="reserved namespace"):
        fiml.FeatureExtractorSpec.from_json(json.dumps(document))


def test_json_and_fluent_extractors_have_identical_names_and_values():
    feature_extractor_spec = configured_spec(capacity=2)
    fluent = fiml.FeatureExtractor(feature_extractor_spec)
    restored = fiml.FeatureExtractor.from_json(feature_extractor_spec.to_json())
    assert restored.feature_names() == fluent.feature_names()

    for extractor in (fluent, restored):
        btc = extractor.symbol("BTCUSDT")
        extractor.update(fiml.KIND_TRADE, btc, 1, price=10.0, volume=1.0)
    np.testing.assert_equal(restored.values(), fluent.values())


def test_vpt_builder_round_trips_and_updates_from_trades():
    spec = fiml.FeatureExtractorSpec().vpt("BTCUSDT")
    document = json.loads(spec.to_json())
    assert document["features"][0]["indicators"] == [{
        "kind": "vpt",
        "source": {"type": "event", "event": "trade"},
    }]

    extractor = fiml.FeatureExtractor.from_json(spec.to_json())
    btc = extractor.symbol("BTCUSDT")
    values = extractor.transform(
        np.full(3, fiml.KIND_TRADE, dtype=np.uint8),
        np.full(3, btc, dtype=np.int64),
        np.arange(3, dtype=np.int64),
        price=np.array([100.0, 110.0, 99.0]),
        volume=np.array([10.0, 20.0, 30.0]),
    )
    np.testing.assert_allclose(values[:, 0], [0.0, 2.0, -1.0])

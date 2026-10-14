import json

import numpy as np
import pandas as pd
import pytest

import fiml


def configured_spec(capacity=4):
    return (fiml.FeatureExtractorSpec(capacity=capacity, checksum="opaque")
            .field("BTCUSDT", source="trade_price", id="price")
            .field("BTCUSDT", source="trade_volume", id="volume"))


def test_fluent_spec_uses_core_json_and_round_trips_metadata():
    feature_extractor_spec = configured_spec()

    assert feature_extractor_spec.capacity == 4
    assert feature_extractor_spec.active_feature_count == 2
    assert feature_extractor_spec.output_count() == 2
    assert feature_extractor_spec.checksum == "opaque"

    document = json.loads(feature_extractor_spec.to_json())
    assert document["version"] == "2.0"
    assert document["capacity"] == 4
    assert document["length"] == 2
    assert document["checksum"] == "opaque"
    assert document["required_events"] == [{"symbol": "btcusdt", "event": "trade"}]
    assert [i["kind"] for i in document["features"][0]["indicators"]] == ["field", "field"]

    restored = fiml.FeatureExtractorSpec.from_json(feature_extractor_spec.to_json())
    assert json.loads(restored.to_json()) == document


@pytest.mark.parametrize("tz", [
    "+-9223372036854775808", "+-1", "+01:-30", "UTC+-0", "++1", "--1",
    "+01:+30", "", " ", "+", "UTC-", "+:30", "+01:", "+01:30:00",
    "01:30", "+15", "-15", "+14:01", "-14:01", "UTC+14:30", "+01:60",
    "+18446744073709551616", "+00:18446744073709551616", "é", "+０１:30",
])
def test_utc_offset_rejects_invalid_text_in_builder_and_json(tz):
    with pytest.raises(ValueError, match="invalid `tz`"):
        fiml.FeatureExtractorSpec().time_since_first_event_of_day(tz)

    spec = fiml.FeatureExtractorSpec().time_since_first_event_of_day()
    document = json.loads(spec.to_json())
    document["features"][0]["indicators"][0]["options"]["utc_offset"] = tz
    with pytest.raises(ValueError, match="invalid UTC offset"):
        fiml.FeatureExtractorSpec.from_json(json.dumps(document))


@pytest.mark.parametrize("tz,millis,canonical", [
    ("UTC", 0, "+00:00"),
    ("+0", 0, "+00:00"),
    ("-00:00", 0, "+00:00"),
    ("UTC+3", 10_800_000, "+03:00"),
    ("UTC-05:30", -19_800_000, "-05:30"),
    ("+01:30", 5_400_000, "+01:30"),
    ("-7", -25_200_000, "-07:00"),
    ("+13:59", 50_340_000, "+13:59"),
    ("+14:00", 50_400_000, "+14:00"),
    ("-14:00", -50_400_000, "-14:00"),
])
def test_utc_offset_preserves_canonical_json_and_day_boundaries(tz, millis, canonical):
    spec = fiml.FeatureExtractorSpec().time_since_first_event_of_day(tz)
    document = json.loads(spec.to_json())
    assert document["features"][0]["indicators"][0]["options"]["utc_offset"] == canonical
    document["features"][0]["indicators"][0]["options"]["utc_offset"] = tz
    restored = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    assert restored.to_json() == spec.to_json()
    assert restored.feature_ids() == spec.feature_ids()

    midnight = 86_400_000 - millis
    for definition in (spec, restored):
        extractor = fiml.FeatureExtractor(definition)
        for timestamp, expected in [
            (midnight - 2_000, 0.0),
            (midnight - 1_000, 1_000.0),
            (midnight, 0.0),
            (midnight + 1_000, 1_000.0),
        ]:
            extractor.update(fiml.KIND_TIME, 0, timestamp)
            np.testing.assert_equal(extractor.values(), [expected])


@pytest.mark.parametrize("tz", [" UTC ", "\tUTC+3\n", "\u00a0-05:30\u00a0"])
def test_utc_offset_whitespace_is_trimmed_only_by_python_builder(tz):
    spec = fiml.FeatureExtractorSpec().time_since_first_event_of_day(tz)
    plain = fiml.FeatureExtractorSpec().time_since_first_event_of_day(tz.strip())
    assert spec.to_json() == plain.to_json()

    document = json.loads(spec.to_json())
    document["features"][0]["indicators"][0]["options"]["utc_offset"] = tz
    with pytest.raises(ValueError, match="invalid UTC offset"):
        fiml.FeatureExtractorSpec.from_json(json.dumps(document))


def test_feature_ids_returns_only_active_stable_ids():
    feature_extractor_spec = configured_spec()

    assert feature_extractor_spec.feature_ids() == fiml.FeatureExtractor(
        feature_extractor_spec
    ).feature_names()[:2]


def test_omitted_capacity_tracks_outputs_and_explicit_capacity_is_fixed():
    dynamic = fiml.FeatureExtractorSpec().field("BTCUSDT").field("BTCUSDT", source="volume").field("BTCUSDT", source="trade_price")
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

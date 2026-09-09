import json
from pathlib import Path

import fiml
import numpy as np
import pytest

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/order_book_replay.json").read_text()
)


def book_event(event):
    payload = {key: value for key, value in event.items() if key != "kind"}
    return getattr(fiml.OrderBookEvent, event["kind"])(**payload)


def snapshot(timestamp=1, update_id=1, size="2", symbol="btcusdt"):
    return fiml.OrderBookEvent.snapshot(
        symbol, timestamp, update_id, [("98", size)], [("102", "2")]
    )


def delta(timestamp, update_id, size, symbol="btcusdt"):
    return fiml.OrderBookEvent.delta(symbol, timestamp, update_id, [("bid", "98", size)])


def runtime(model, buffer_size=8):
    spec = (fiml.FeatureExtractorSpec()
            .configure_order_book("BTCUSDT", update_policy="contiguous", buffer_size=buffer_size)
            .order_book_best_bid_size("BTCUSDT"))
    if model:
        spec = fiml.PipelineSpec(spec).identity(spec.feature_ids()[0], output="bid_size")
        return fiml.ModelInputPipeline(spec)
    return fiml.FeatureExtractor(spec)


@pytest.mark.parametrize("model", [False, True])
def test_shared_book_fixture_matches_rust_exactly(model):
    document = FIXTURE["pipeline"] if model else FIXTURE["pipeline"]["feature_extractor"]
    cls = fiml.ModelInputPipeline if model else fiml.FeatureExtractor
    instance = cls.from_json(json.dumps(document))
    assert instance.feature_names() == FIXTURE["model_names" if model else "raw_names"]
    assert np.isnan(instance.values()).all()
    if model:
        assert instance.raw_feature_names() == FIXTURE["raw_names"]
        assert np.isnan(instance.raw_values()).all()
    for step in FIXTURE["steps"]:
        event = step["event"]
        def apply():
            if event["kind"] in ("snapshot", "delta"):
                instance.update_order_book(book_event(event))
            elif event["kind"] == "time":
                instance.update(fiml.KIND_TIME, 0, event["timestamp"])
            else:
                instance.update(fiml.KIND_PRICE, instance.symbol(event["symbol"]),
                                event["timestamp"], price=event["price"])
        if step["error"]:
            with pytest.raises(ValueError, match=step["error"]):
                apply()
        else:
            apply()
        np.testing.assert_equal(instance.values(), np.array(step["model" if model else "raw"], dtype=float))
        if model:
            np.testing.assert_equal(instance.raw_values(), np.array(step["raw"], dtype=float))
    assert np.isnan(cls.from_json(json.dumps(document)).values()).all()


def test_all_book_builders_preserve_configuration_and_default_ids():
    document = json.loads(json.dumps(FIXTURE["pipeline"]["feature_extractor"]))
    spec = fiml.FeatureExtractorSpec(capacity=21)
    for config in reversed(document["order_books"]):
        spec.configure_order_book(config["symbol"].upper(), update_policy=config["update_policy"],
                                  buffer_size=config["buffer_size"])
    for group in document["features"]:
        for indicator in group["indicators"]:
            method = getattr(spec, indicator["kind"])
            if indicator["kind"] == "order_book_imbalance":
                method(group["symbol"], [output["n_levels"] for output in indicator["outputs"]])
                for output in indicator["outputs"]:
                    del output["id"]
            else:
                method(group["symbol"], **indicator.get("options", {}))
                del indicator["outputs"]
    expected = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    assert spec.to_json() == expected.to_json()
    assert spec.feature_ids() == expected.feature_ids()
    assert spec.indicator_count() == 19
    assert fiml.FeatureExtractor(spec).active_feature_count() == 20


@pytest.mark.parametrize("model", [False, True])
def test_book_batches_prevalidate_then_preserve_sequence_error_prefix(model):
    instance = runtime(model)
    for events, message in [
        ([snapshot(), delta(2, 2, "4", "missing")], "row 1"),
        ([snapshot(), delta(0, 2, "4")], "row 1"),
    ]:
        with pytest.raises(ValueError, match=message):
            instance.transform_order_book(events)
        assert np.isnan(instance.values()).all()
        instance.output_dtype = "float32"  # No accepted event has locked dtype.
        instance.output_dtype = "float64"
    with pytest.raises(TypeError):
        instance.transform_order_book([snapshot(), object()])
    assert np.isnan(instance.values()).all()
    np.testing.assert_equal(instance.transform_order_book([snapshot(), delta(2, 2, "4")]), [[2], [4]])
    with pytest.raises(ValueError, match="row 1.*expected update ID 4, received 5"):
        instance.transform_order_book([delta(3, 3, "6"), delta(4, 5, "8"), delta(5, 6, "10")])
    np.testing.assert_equal(instance.values(), [6])
    # The rejected gap delta was retained; the later batch row was not applied.
    instance.update_order_book(snapshot(4, 4))
    np.testing.assert_equal(instance.values(), [8])
    instance.update_order_book(delta(5, 6, "10"))
    np.testing.assert_equal(instance.values(), [10])
    assert instance.transform_order_book([]).shape == (0, 1)
    with pytest.raises(ValueError, match="row 0"):
        instance.transform_order_book([delta(4, 7, "12")])
    np.testing.assert_equal(instance.values(), [10])


@pytest.mark.parametrize("model", [False, True])
def test_book_batches_validate_timestamps_per_symbol(model):
    spec = fiml.FeatureExtractorSpec()
    for symbol in ["btcusdt", "ethusdt"]:
        spec.configure_order_book(symbol, update_policy="contiguous", buffer_size=8)
        spec.order_book_best_bid_size(symbol)
    if model:
        pipeline_spec = fiml.PipelineSpec(spec)
        for feature_id in spec.feature_ids():
            pipeline_spec.identity(feature_id)
        instance = fiml.ModelInputPipeline(pipeline_spec)
    else:
        instance = fiml.FeatureExtractor(spec)

    rows = instance.transform_order_book([
        snapshot(100, 1), snapshot(-100, 1, symbol="ethusdt"),
        delta(101, 2, "4"), delta(-99, 2, "6", "ethusdt"),
    ])
    np.testing.assert_equal(rows, [[2, np.nan], [2, 2], [4, 2], [4, 6]])
    with pytest.raises(ValueError, match="row 1.*previous timestamp 101"):
        instance.transform_order_book([
            delta(-98, 3, "8", "ethusdt"), delta(100, 3, "10"),
        ])
    np.testing.assert_equal(instance.values(), [4, 6])
    # The valid prefix was not applied when another symbol failed prevalidation.
    instance.update_order_book(delta(-99, 3, "8", "ethusdt"))
    np.testing.assert_equal(instance.values(), [4, 8])


@pytest.mark.parametrize("model", [False, True])
def test_capacity_error_requires_snapshot_and_does_not_change_values(model):
    instance = runtime(model, buffer_size=1)
    instance.update_order_book(snapshot())
    instance.update_order_book(delta(2, 2, "4"))
    with pytest.raises(ValueError, match="capacity 1 was exceeded"):
        instance.update_order_book(delta(3, 3, "6"))
    np.testing.assert_equal(instance.values(), [4])
    instance.update_order_book(snapshot(3, 3, "5"))
    instance.update_order_book(delta(4, 4, "7"))
    np.testing.assert_equal(instance.values(), [7])


def test_exact_decimal_levels_do_not_collapse_to_float_prices():
    price = "0.1000000000000000000000000001"
    spec = (fiml.FeatureExtractorSpec()
            .configure_order_book("btc", update_policy="monotonic", buffer_size=2)
            .order_book_level_size("btc", "bid", "0.1")
            .order_book_level_size("btc", "bid", price))
    instance = fiml.FeatureExtractor(spec)
    instance.update_order_book(fiml.OrderBookEvent.snapshot("btc", 0, 1,
                                [("0.1", "2"), (price, "3")], []))
    np.testing.assert_equal(instance.values(), [2, 3])
    instance.update_order_book(fiml.OrderBookEvent.delta("btc", 1, 2, [("bid", price, "0")]))
    np.testing.assert_equal(instance.values(), [2, np.nan])


@pytest.mark.parametrize("value", [0.1, float("nan"), float("inf"), "NaN", "-1",
                                      "0.12345678901234567890123456789", "1e100"])
def test_event_constructors_reject_inexact_or_invalid_decimals(value):
    for constructor in [
        lambda: fiml.OrderBookEvent.snapshot("btc", 0, 1, [(value, "2")], []),
        lambda: fiml.OrderBookEvent.delta("btc", 0, 1, [("bid", "1", value)]),
    ]:
        with pytest.raises((ValueError, TypeError)):
            constructor()


def test_event_constructors_validate_ids_timestamps_and_sides():
    for args in [(0, -1), (0, 2**64), (-2**63-1, 0), (2**63, 0)]:
        with pytest.raises(OverflowError):
            snapshot(*args)
    with pytest.raises(ValueError, match="side"):
        fiml.OrderBookEvent.delta("btc", 0, 1, [("buy", "1", "2")])
    instance = runtime(False)
    instance.update_order_book(snapshot(-2**63, 2**64-1))
    instance.update_order_book(delta(2**63-1, 2**64-1, "9"))
    np.testing.assert_equal(instance.values(), [2])


@pytest.mark.parametrize("model", [False, True])
def test_legacy_placeholder_fails_before_numeric_batch_mutation(model):
    instance = runtime(model)
    with pytest.raises(ValueError, match="use OrderBookEvent"):
        instance.update(fiml.KIND_ORDERBOOK, 0, 0, bid=98.0, ask=102.0)
    with pytest.raises(ValueError, match="row 1.*use OrderBookEvent"):
        instance.transform(np.array([fiml.KIND_TIME, fiml.KIND_ORDERBOOK], dtype=np.uint8),
                           np.array([0, 0], dtype=np.int64), np.array([0, 1], dtype=np.int64))
    instance.output_dtype = "float32"
    assert np.isnan(instance.values()).all()


def test_missing_duplicate_and_global_configuration_fail_before_replay():
    spec = fiml.FeatureExtractorSpec().order_book_mid_price("btc")
    with pytest.raises(ValueError, match="order book"):
        fiml.FeatureExtractor(spec)
    with pytest.raises(ValueError, match="order book"):
        fiml.ModelInputPipeline(fiml.PipelineSpec(spec).identity(spec.feature_ids()[0], output="mid"))
    spec.configure_order_book("btc", update_policy="contiguous", buffer_size=1)
    before = spec.to_json()
    for symbol, policy in [("BTC", "contiguous"), ("__global__", "monotonic"), ("eth", "invalid")]:
        with pytest.raises(ValueError):
            spec.configure_order_book(symbol, update_policy=policy, buffer_size=1)
        assert spec.to_json() == before

from pipeline_fixtures import LinearSpec, SingleOutput
import json
from pathlib import Path

import fiml
import numpy as np
import pytest

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/order_book_replay.json").read_text()
)

FLOW_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/order_book_flow.json").read_text()
)


def book_event(event):
    payload = {key: value for key, value in event.items() if key != "kind"}
    constructor = "delta_range" if event["kind"] == "delta" else event["kind"]
    return getattr(fiml.OrderBookEvent, constructor)(**payload)


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
        spec = LinearSpec(spec).identity(spec.feature_ids()[0], output="bid_size")
        return SingleOutput(spec)
    return fiml.FeatureExtractor(spec)


@pytest.mark.parametrize("fixture", [FIXTURE, FLOW_FIXTURE])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("model", [False, True])
def test_shared_book_fixture_matches_rust_exactly(model, dense, fixture):
    document = fixture["pipeline"] if model else fixture["pipeline"]["nodes"][0]["spec"]
    document = json.loads(json.dumps(document))
    if dense:
        raw = document["nodes"][0]["spec"] if model else document
        for config in raw["order_books"]:
            config["dense"] = {"tick_size": "1", "min_price": "0", "max_price": "1000"}
    cls = SingleOutput if model else fiml.FeatureExtractor
    instance = cls.from_json(json.dumps(document))
    assert instance.feature_names() == fixture["model_names" if model else "raw_names"]
    assert np.isnan(instance.values()).all()
    if model:
        assert instance.raw_feature_names() == fixture["raw_names"]
        assert np.isnan(instance.raw_values()).all()
    for step in fixture["steps"]:
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
            with pytest.raises(ValueError, match=step.get("python_error", step["error"])):
                apply()
        else:
            apply()
        np.testing.assert_equal(instance.values(), np.array(step["model" if model else "raw"], dtype=float))
        if model:
            np.testing.assert_equal(instance.raw_values(), np.array(step["raw"], dtype=float))
    assert np.isnan(cls.from_json(json.dumps(document)).values()).all()


def test_all_book_builders_preserve_configuration_and_default_ids():
    document = json.loads(json.dumps(FIXTURE["pipeline"]["nodes"][0]["spec"]))
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
        pipeline_spec = LinearSpec(spec)
        for feature_id in spec.feature_ids():
            pipeline_spec.identity(feature_id)
        instance = SingleOutput(pipeline_spec)
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
def test_ranges_apply_once_overlap_and_replay_after_snapshot(model):
    instance = runtime(model)
    rows = instance.transform_order_book([
        fiml.OrderBookEvent.delta_range("BTCUSDT", 0, 101, 105, [("bid", "98", "5")]),
        snapshot(1, 103),
        fiml.OrderBookEvent.delta_range(
            symbol="BTCUSDT", timestamp=2, first_update_id=106, final_update_id=110,
            changes=[("bid", "98", "7")]),
        fiml.OrderBookEvent.delta_range("BTCUSDT", 3, 109, 115, [("bid", "98", "9")]),
        fiml.OrderBookEvent.delta_range("BTCUSDT", 4, 101, 110, [("bid", "98", "99")]),
        fiml.OrderBookEvent.delta_range("BTCUSDT", 5, 109, 115, [("bid", "98", "99")]),
        snapshot(6, 108),
        fiml.OrderBookEvent.delta("BTCUSDT", 7, 116, [("bid", "98", "11")]),
    ])
    np.testing.assert_equal(rows, [[np.nan], [5], [7], [9], [9], [9], [9], [11]])


@pytest.mark.parametrize("model", [False, True])
def test_range_gap_retains_delta_until_snapshot_covers_missing_history(model):
    instance = runtime(model)
    instance.update_order_book(snapshot(0, 100))
    with pytest.raises(ValueError, match="expected update ID 101, received 102"):
        instance.update_order_book(fiml.OrderBookEvent.delta_range(
            "BTCUSDT", 1, 102, 105, [("bid", "98", "5")]))
    np.testing.assert_equal(instance.values(), [2])
    with pytest.raises(ValueError, match="expected update ID 106, received 107"):
        instance.update_order_book(fiml.OrderBookEvent.delta_range(
            "BTCUSDT", 2, 107, 110, [("bid", "98", "7")]))
    with pytest.raises(ValueError, match="snapshot 103 requires update ID 106, received 107"):
        instance.update_order_book(snapshot(3, 103))
    np.testing.assert_equal(instance.values(), [2])
    instance.update_order_book(snapshot(3, 108))
    np.testing.assert_equal(instance.values(), [7])
    instance.update_order_book(delta(4, 111, "9"))
    np.testing.assert_equal(instance.values(), [9])


@pytest.mark.parametrize("model", [False, True])
def test_range_construction_rejects_invalid_ids_before_runtime_changes(model):
    instance = runtime(model)
    instance.update_order_book(snapshot(0, 100))
    for first, final_id in [(106, 105), (101, 99), (2**64 - 1, 0)]:
        with pytest.raises(ValueError, match="invalid update ID range"):
            instance.update_order_book(fiml.OrderBookEvent.delta_range(
                "BTCUSDT", 1, first, final_id, [("bid", "98", "99")]))
    for first, final_id in [(-1, 0), (0, -1), (2**64, 2**64 - 1), (0, 2**64)]:
        with pytest.raises(OverflowError):
            fiml.OrderBookEvent.delta_range("BTCUSDT", 1, first, final_id, [])
    np.testing.assert_equal(instance.values(), [2])
    instance.update_order_book(fiml.OrderBookEvent.delta_range(
        "BTCUSDT", 1, 101, 105, [("bid", "98", "5")]))
    np.testing.assert_equal(instance.values(), [5])


@pytest.mark.parametrize("model", [False, True])
def test_range_replay_supports_maximum_ids(model):
    instance = runtime(model)
    maximum = 2**64 - 1
    rows = instance.transform_order_book([
        fiml.OrderBookEvent.delta_range("BTCUSDT", 0, maximum - 4, maximum,
                                       [("bid", "98", "7")]),
        snapshot(1, maximum - 2),
        fiml.OrderBookEvent.delta_range("BTCUSDT", 2, 0, maximum,
                                       [("bid", "98", "99")]),
        snapshot(3, maximum, "3"),
        delta(4, maximum, "99"),
    ])
    np.testing.assert_equal(rows, [[np.nan], [7], [7], [3], [3]])


@pytest.mark.parametrize("model", [False, True])
def test_legacy_placeholder_fails_before_numeric_batch_mutation(model):
    instance = runtime(model)
    with pytest.raises(ValueError, match="use OrderBookEvent"):
        instance.update(fiml.KIND_ORDERBOOK, 0, 0)
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
        SingleOutput(LinearSpec(spec).identity(spec.feature_ids()[0], output="mid"))
    spec.configure_order_book("btc", update_policy="contiguous", buffer_size=1)
    before = spec.to_json()
    for symbol, policy in [("BTC", "contiguous"), ("__global__", "monotonic"), ("eth", "invalid")]:
        with pytest.raises(ValueError):
            spec.configure_order_book(symbol, update_policy=policy, buffer_size=1)
        assert spec.to_json() == before


@pytest.mark.parametrize("model", [False, True])
def test_dense_json_configuration_selects_storage_and_preserves_grid(model):
    from jsonschema import Draft202012Validator

    spec = (fiml.FeatureExtractorSpec()
            .configure_order_book("btc", update_policy="contiguous", buffer_size=4)
            .order_book_best_bid_price("btc"))
    document = json.loads(spec.to_json())
    grid = {"tick_size": "0.01", "min_price": "99", "max_price": "102"}
    document["order_books"][0]["dense"] = grid
    schema = json.loads((Path(__file__).parents[3] / "docs/feature-extractor-spec.schema.json").read_text())
    validator = Draft202012Validator(schema)
    validator.validate(document)
    restored = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    assert json.loads(restored.to_json())["order_books"][0]["dense"] == grid
    instance = (SingleOutput(LinearSpec(restored).identity(restored.feature_ids()[0]))
                if model else fiml.FeatureExtractor(restored))
    instance.update_order_book(fiml.OrderBookEvent.snapshot("btc", 0, 0, [("100", "2")], []))
    instance.update_order_book(fiml.OrderBookEvent.delta("btc", 1, 1, [("bid", "101", "3")]))
    np.testing.assert_equal(instance.values(), [101])
    for price in ["101.001", "103"]:
        with pytest.raises(ValueError, match="invalid Bid level"):
            instance.update_order_book(fiml.OrderBookEvent.delta("btc", 2, 2, [("bid", price, "1")]))
        np.testing.assert_equal(instance.values(), [101])
    for invalid in [None, {}, {**grid, "tick_size": 0.01}, {**grid, "tick_size": "0"}, {**grid, "extra": 1}]:
        document["order_books"][0]["dense"] = invalid
        assert list(validator.iter_errors(document))
        with pytest.raises(ValueError):
            fiml.FeatureExtractorSpec.from_json(json.dumps(document))


def test_flow_builders_roundtrip_and_feed_scalar_averages():
    from test_pipeline_spec_schema import VALIDATOR

    raw = (fiml.FeatureExtractorSpec()
           .configure_order_book("BTC", update_policy="contiguous", buffer_size=16)
           .order_book_best_bid_size("BTC")
           .order_book_bid_added_volume("BTC")
           .order_book_bid_removed_volume("BTC")
           .order_book_ask_added_volume("BTC")
           .order_book_ask_removed_volume("BTC"))
    names = raw.feature_ids()
    assert names == [f"{kind}:symbol=3:btc:source=order_book" for kind in [
        "order_book_best_bid_size", "order_book_bid_added_volume",
        "order_book_bid_removed_volume", "order_book_ask_added_volume",
        "order_book_ask_removed_volume",
    ]]
    spec = (LinearSpec(raw)
            .sma(names[1], window=2, warmup=fiml.WarmupPolicy.FIRST_VALUE, output="bid_added_sma")
            .ema(names[4], window=3, warmup=fiml.WarmupPolicy.FIRST_VALUE, output="ask_removed_ema"))
    document = json.loads(spec.to_json())
    VALIDATOR.validate(document)
    VALIDATOR.validate(FLOW_FIXTURE["pipeline"])
    assert json.loads(LinearSpec.from_json(spec.to_json()).to_json()) == document
    runtime = SingleOutput(spec)
    # The first buffered update and snapshot must not seed the sample averages.
    events = [book_event(step["event"]) for step in FLOW_FIXTURE["steps"][:7]]
    actual = runtime.transform_order_book(events)
    np.testing.assert_equal(actual, np.array([s["model"] for s in FLOW_FIXTURE["steps"][:7]], dtype=float))
    runtime.reset()
    chunks = [runtime.transform_order_book(events[:2]), runtime.transform_order_book(events[2:])]
    np.testing.assert_equal(np.concatenate(chunks), actual)


@pytest.mark.parametrize("fixture", [FIXTURE, FLOW_FIXTURE])
@pytest.mark.parametrize("dense", [False, True])
def test_branching_book_graph_matches_shared_oracles(fixture, dense):
    document = json.loads(json.dumps(fixture["pipeline"]))
    if dense:
        for config in document["nodes"][0]["spec"]["order_books"]:
            config["dense"] = {"tick_size": "1", "min_price": "0", "max_price": "1000"}
    document["nodes"].extend([
        {"type": "transform", "parent": 1, "stage": {"type": "scalar", "transformations": [
            {"type": "standard_scale", "input": fixture["model_names"][0],
             "output": "scaled", "mean": 1., "scale": 2.},
        ]}},
        {"type": "transform", "parent": 1, "stage": {"type": "scalar", "transformations": [
            {"type": "identity", "input": fixture["model_names"][-1], "output": "selected"},
        ]}},
    ])
    document["outputs"] = [
        {"name": "model_a", "node": 2}, {"name": "decision", "node": 0},
        {"name": "shared", "node": 1}, {"name": "model_b", "node": 3},
        {"name": "alias", "node": 1},
    ]
    instance = fiml.ModelInputPipeline.from_json(json.dumps(document))
    assert instance.output_names() == [output["name"] for output in document["outputs"]]
    for step in fixture["steps"]:
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
            with pytest.raises(ValueError, match=step.get("python_error", step["error"])):
                apply()
        else:
            apply()
        raw, model = np.array(step["raw"], dtype=float), np.array(step["model"], dtype=float)
        expected = {"model_a": (model[:1] - 1.) / 2., "model_b": model[-1:],
                    "shared": model, "alias": model,
                    "decision": raw[:document["nodes"][0]["spec"]["length"]]}
        np.testing.assert_equal(instance.raw_values(), raw)
        for name in instance.output_names():
            np.testing.assert_equal(instance.output(name).values(), expected[name])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_branching_book_batch_context_prevalidation_and_error_prefix(dtype):
    raw = (fiml.FeatureExtractorSpec()
           .configure_order_book("btcusdt", update_policy="contiguous", buffer_size=8)
           .order_book_best_bid_size("btcusdt").context("btcusdt", "high", id="high"))
    size_id = next(name for name in raw.feature_ids() if name != "high")
    spec = fiml.PipelineSpec(raw)
    shared = spec.add(spec.root(), fiml.ScalarStage().identity(size_id, output="size")
                      .identity("high"))
    spec.output("model_a", spec.add(shared, fiml.ScalarStage()
                                    .sma("size", window=2, warmup=fiml.WarmupPolicy.FULL_WINDOW)))
    spec.output("model_b", spec.add(shared, fiml.ScalarStage().lagged("size", lag_window=1)))
    spec.output("decision", spec.root())
    spec.output("shared", shared)
    batch, manual = spec.build(dtype), spec.build(dtype)

    for invalid, schedule in [
        ([snapshot(), delta(2, 2, "4", "missing")], {0: {"high": 100.}}),
        ([snapshot(), delta(0, 2, "4")], {0: {"high": 100.}}),
        ([snapshot(), delta(2, 2, "4")], {0: {"high": 100.}, 1: {"missing": 1.}}),
    ]:
        with pytest.raises(ValueError, match="row 1"):
            batch.transform_order_book(invalid, context_updates=schedule)
        assert np.isnan(batch.raw_values()).all()
        for name in batch.output_names():
            assert np.isnan(batch.output(name).values()).all()
        batch.output_dtype = dtype

    initial = [delta(0, 2, "4"), snapshot(1, 1), delta(2, 2, "99"), delta(3, 3, "6")]
    schedule = {0: {"high": 100.}, 2: {"high": 200.}}
    actual = batch.transform_order_book(initial, context_updates=schedule)
    assert list(actual) == batch.output_names()
    for row, event in enumerate(initial):
        if row in schedule:
            manual.update_context(schedule[row])
        manual.update_order_book(event)
        for name, matrix in actual.items():
            assert matrix.dtype == np.dtype(dtype)
            np.testing.assert_equal(matrix[row], manual.output(name).values())
    np.testing.assert_equal(actual["model_a"], [[np.nan], [np.nan], [np.nan], [5.]])
    np.testing.assert_equal(actual["model_b"], [[np.nan], [np.nan], [4.], [4.]])
    np.testing.assert_equal(actual["shared"], [[np.nan, 100.], [4., 100.], [4., 200.], [6., 200.]])

    failing = [delta(4, 4, "8"), delta(5, 6, "12"), delta(6, 7, "14")]
    with pytest.raises(ValueError, match="row 1.*expected update ID 5, received 6"):
        batch.transform_order_book(failing, context_updates={0: {"high": 300.}, 2: {"high": 999.}})
    manual.update_context({"high": 300.})
    manual.update_order_book(failing[0])
    with pytest.raises(ValueError, match="expected update ID 5, received 6"):
        manual.update_order_book(failing[1])
    for name in batch.output_names():
        np.testing.assert_equal(batch.output(name).values(), manual.output(name).values())
    np.testing.assert_equal(batch.output("model_a").values(), [7.])
    np.testing.assert_equal(batch.output("model_b").values(), [6.])
    np.testing.assert_equal(batch.output("shared").values(), [8., 300.])

    # Resynchronization must replay the rejected delta, but not the skipped row.
    recovery = [snapshot(5, 5), failing[2]]
    recovered = batch.transform_order_book(recovery)
    for row, event in enumerate(recovery):
        manual.update_order_book(event)
        for name in recovered:
            np.testing.assert_equal(recovered[name][row], manual.output(name).values())
    np.testing.assert_equal(recovered["shared"], [[12., 300.], [14., 300.]])
    np.testing.assert_equal(recovered["model_a"], [[10.], [13.]])
    np.testing.assert_equal(recovered["model_b"], [[8.], [12.]])
    empty = batch.transform_order_book([])
    assert list(empty) == batch.output_names()
    for name, matrix in empty.items():
        assert matrix.shape == (0, batch.output(name).n_features())
        assert matrix.dtype == np.dtype(dtype)

import json
from pathlib import Path

import fiml
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from sklearn.preprocessing import StandardScaler

SCHEMA = json.loads((Path(__file__).parents[3] / "docs/feature-extractor-spec.schema.json").read_text())


def raw_spec():
    return (fiml.FeatureExtractorSpec()
            .field("btc", id="price")
            .context("btc", "previous_day_high", id="high")
            .context("__global__", "previous_day_volume", id="volume"))


def model_spec():
    return (fiml.PipelineSpec(raw_spec())
            .identity("high").lagged("high", lag_window=1, output="lag")
            .sma("price", window=2, warmup=fiml.WarmupPolicy.FULL_WINDOW, output="sma")
            .ema("high", window=1, output="context_ema"))


def events(runtime):
    symbol = runtime.symbol("btc")
    return dict(kind=np.full(4, fiml.KIND_PRICE, dtype=np.uint8),
                symbol=np.full(4, symbol, dtype=np.int64),
                timestamp=np.arange(4, dtype=np.int64), price=np.array([10., 20., 30., 40.]))


@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scheduled_replay_matches_manual_interleaving(model, dtype):
    cls = fiml.ModelInputPipeline if model else fiml.FeatureExtractor
    spec = model_spec() if model else raw_spec()
    manual, batch = cls(spec, dtype), cls(spec, dtype)
    data = events(batch)
    events(manual)
    schedule = {0: {"high": 100., "volume": 1000.}, 2: {"high": 300.}, 3: {"volume": None}}
    expected = []
    for row in range(4):
        if row in schedule:
            manual.update_context(schedule[row])
        manual.update(fiml.KIND_PRICE, 0, row, price=data["price"][row])
        expected.append(manual.values())
    actual = batch.transform(**data, context_updates=schedule)
    np.testing.assert_equal(actual, expected)
    assert actual.dtype == np.dtype(dtype)
    if model:
        np.testing.assert_equal(actual, [[100., np.nan, np.nan, np.nan],
                                         [100., 100., 15., np.nan],
                                         [300., 100., 25., np.nan],
                                         [300., 300., 35., np.nan]])
        batch.reset()
        assert np.isnan(batch.raw_values()).all()
        np.testing.assert_equal(batch.transform(**data, context_updates=schedule), actual)
    # A call's row indices restart at zero; held context survives between calls.
    later = {**data, "timestamp": data["timestamp"] + 4}
    batch.transform(**later, context_updates={0: {"high": 400.}})
    assert batch.values()[batch.feature_names().index("high")] == 400.


@pytest.mark.parametrize("model", [False, True])
def test_invalid_schedule_is_rejected_before_any_event_or_context_change(model):
    cls = fiml.ModelInputPipeline if model else fiml.FeatureExtractor
    runtime = cls(model_spec() if model else raw_spec())
    data = events(runtime)
    for schedule in [{-1: {}}, {4: {}}, {1.5: {}}, {True: {}}, {"0": {}},
                     {0: {"high": 10.}, 3: {"missing": 2.}},
                     {0: {"high": 10.}, 3: {"price": 2.}},
                     {0: {"high": 10.}, 3: {"high": np.inf}},
                     {0: {"high": np.nan}}, {0: {"high": -np.inf}},
                     {0: {1: 2.}}, {0: []}]:
        with pytest.raises((ValueError, TypeError)):
            runtime.transform(**data, context_updates=schedule)
        assert np.isnan(runtime.values()).all()
        runtime.output_dtype = "float32"
        runtime.output_dtype = "float64"
    with pytest.raises(ValueError):
        runtime.transform(**{**data, "timestamp": np.array([1, 0, 2, 3], dtype=np.int64)},
                          context_updates={0: {"high": 10.}})
    assert np.isnan(runtime.values()).all()
    runtime.update_context({})
    runtime.output_dtype = "float32"
    with pytest.raises(ValueError):
        runtime.update_context({"high": 10., "price": 20.})
    assert np.isnan(runtime.values()).all()
    runtime.update_context({"high": 10.})
    with pytest.raises(ValueError):
        runtime.output_dtype = "float64"
    # Context did not advance the timestamp watermark.
    runtime.update(fiml.KIND_PRICE, 0, -100, price=1.)


def test_context_update_locks_recipe_and_reset_clears_context():
    pipeline = fiml.ModelInputPipeline(fiml.PipelineSpec(raw_spec()).identity("high"))
    pipeline.update_context({"high": 5.})
    with pytest.raises(ValueError, match="recipe"):
        pipeline.add_transformation(StandardScaler(), name="scale")
    assert pipeline.values()[0] == 5.
    document = pipeline.to_json()
    restored = fiml.ModelInputPipeline.from_json(document)
    assert np.isnan(restored.values()).all()
    assert np.isnan(restored.raw_values()).all()
    pipeline.reset()
    assert np.isnan(pipeline.raw_values()).all()
    with pytest.raises(ValueError, match="recipe"):
        pipeline.add_transformation(StandardScaler(), name="scale")


def test_fit_reapplies_schedule_at_each_stage_and_preserves_failed_refit_state():
    spec = fiml.PipelineSpec(raw_spec()).identity("high")
    pipeline = (fiml.ModelInputPipeline(spec)
                .add_transformation(StandardScaler(), name="first")
                .add_transformation(fiml.ScalarStage().lagged("high", lag_window=1, output="lag"), name="history")
                .add_transformation(StandardScaler(), name="second"))
    data = events(pipeline)
    schedule = {0: {"high": 10.}, 1: {"high": 20.}, 2: {"high": 30.}}
    fit_mask = np.array([False, True, False, True])
    first = StandardScaler().fit(np.array([[20.], [30.]]))
    base = first.transform(np.array([[10.], [20.], [30.], [30.]]))
    lagged = np.vstack(([np.nan], base[:-1]))
    second = StandardScaler().fit(lagged[fit_mask])
    actual = pipeline.fit_transform(**data, context_updates=schedule, fit_mask=fit_mask)
    np.testing.assert_allclose(actual, second.transform(lagged), equal_nan=True)
    before = pipeline.values().copy()
    artifact = pipeline.to_json()
    for bad in [{0: {"missing": 10.}}, {0: {"high": None}}]:
        with pytest.raises(ValueError):
            pipeline.fit(**data, context_updates=bad, fit_mask=fit_mask)
        np.testing.assert_equal(pipeline.values(), before)
        assert pipeline.to_json() == artifact
    pipeline.fit(**data, context_updates=schedule, fit_mask=fit_mask)
    assert np.isnan(pipeline.raw_values()).all()
    np.testing.assert_allclose(pipeline.transform(**data, context_updates=schedule), actual, equal_nan=True)


@pytest.mark.parametrize("model", [False, True])
def test_book_replay_retains_context_and_prevalidates_schedule(model):
    spec = (fiml.FeatureExtractorSpec().configure_order_book("btc", update_policy="contiguous", buffer_size=8)
            .order_book_best_bid_price("btc").context("btc", "high", id="high"))
    names = spec.feature_ids()
    runtime = (fiml.ModelInputPipeline(fiml.PipelineSpec(spec).identity("high")) if model
               else fiml.FeatureExtractor(spec))
    book_events = [fiml.OrderBookEvent.snapshot("btc", 1, 1, [("100", "1")], []),
                   fiml.OrderBookEvent.delta("btc", 2, 2, [("bid", "101", "1")])]
    with pytest.raises(ValueError):
        runtime.transform_order_book(book_events, context_updates={0: {"high": 1.}, 1: {"missing": 1.}})
    assert np.isnan(runtime.values()).all()
    matrix = runtime.transform_order_book(book_events, context_updates={0: {"high": 105.}})
    column = 0 if model else names.index("high")
    np.testing.assert_equal(matrix[:, column], [105., 105.])
    runtime.update_context({"high": 110.})
    if not model:
        assert runtime.values()[names.index(next(name for name in names if name != "high"))] == 101.


def test_context_json_schema_names_and_canonical_order():
    first = (fiml.FeatureExtractorSpec().context("btc", "high", id="high")
             .context("__global__", "volume").context("eth", "high", id="eth_high"))
    second = (fiml.FeatureExtractorSpec().context("eth", "high", id="eth_high")
              .context("__global__", "volume").context("btc", "high", id="high"))
    assert first.to_json() == second.to_json()
    assert first.indicator_count() == 3
    document = json.loads(first.to_json())
    assert document["required_events"] == []
    assert document["version"] == "2.0"
    Draft202012Validator(SCHEMA).validate(document)
    assert fiml.FeatureExtractorSpec.from_json(first.to_json()).to_json() == first.to_json()
    for mutation in [lambda i: i.update(source={"type": "any_event"}),
                     lambda i: i.update(source={"type": "context", "event": "time"}),
                     lambda i: i.update(options={"name": ""}),
                     lambda i: i.update(options={"name": "x", "aggregation": "1s"}),
                     lambda i: i.update(warmup_policy="first_value"),
                     lambda i: i.update(outputs=[{"lag": 1}]),
                     lambda i: i.update(outputs=[{"id": "a"}, {"id": "b"}]),
                     lambda i: i.update(outputs=[{"id": "__reserved_0"}]),
                     lambda i: i.pop("options")]:
        invalid = json.loads(first.to_json())
        mutation(invalid["features"][0]["indicators"][0])
        assert list(Draft202012Validator(SCHEMA).iter_errors(invalid))
        with pytest.raises(ValueError):
            fiml.FeatureExtractorSpec.from_json(json.dumps(invalid))
    for spec in [fiml.FeatureExtractorSpec().context("btc", ""),
                 fiml.FeatureExtractorSpec().context("btc", "high").context("btc", "high"),
                 fiml.FeatureExtractorSpec().context("btc", "high", id="x").context("eth", "high", id="x")]:
        with pytest.raises(ValueError):
            fiml.FeatureExtractor(spec)


def test_shared_rust_python_context_replay_fixture():
    fixture = json.loads((Path(__file__).parents[3] / "tests/fixtures/context_replay.json").read_text())
    runtime = fiml.ModelInputPipeline.from_json(json.dumps(fixture["pipeline"]))
    steps = fixture["steps"]
    handle = runtime.symbol("btc")
    actual = runtime.transform(
        np.full(len(steps), fiml.KIND_PRICE, dtype=np.uint8),
        np.full(len(steps), handle, dtype=np.int64),
        np.array([step["timestamp"] for step in steps], dtype=np.int64),
        price=np.array([step["price"] for step in steps]),
        context_updates={row: step["context"] for row, step in enumerate(steps)},
    )
    np.testing.assert_equal(actual, np.array([step["expected"] for step in steps], dtype=float))
    from test_pipeline_spec_schema import VALIDATOR
    VALIDATOR.validate(fixture["pipeline"])

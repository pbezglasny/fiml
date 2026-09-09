import json
from pathlib import Path

import pytest
import fiml
from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).parents[3] / "docs" / "feature-extractor-spec.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def document_with_source(source):
    return {
        "version": "1.0",
        "capacity": 1,
        "length": 1,
        "features": [
            {
                "symbol": "BTCUSDT",
                "indicators": [
                    {
                        "kind": "sma",
                        "source": source,
                        "warmup_policy": "first_value",
                        "outputs": [{"window": 1}],
                    }
                ],
            }
        ],
    }


def assert_valid_source(source):
    assert not list(VALIDATOR.iter_errors(document_with_source(source)))


def assert_invalid_source(source):
    assert list(VALIDATOR.iter_errors(document_with_source(source)))


def test_feature_extractor_spec_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize(
    ("event", "field"),
    [
        ("price", "value"),
        ("volume", "value"),
        ("trade", "price"),
        ("trade", "volume"),
    ],
)
def test_schema_accepts_supported_field_sources(event, field):
    assert_valid_source({"type": "field", "event": event, "field": field})


@pytest.mark.parametrize(
    ("event", "field"),
    [
        ("price", "price"),
        ("price", "volume"),
        ("volume", "price"),
        ("volume", "volume"),
        ("trade", "value"),
    ],
)
def test_schema_rejects_unsupported_field_sources(event, field):
    assert_invalid_source({"type": "field", "event": event, "field": field})


@pytest.mark.parametrize(
    "event",
    ["price", "volume", "trade", "order_book_delta", "order_book_snapshot", "time"],
)
def test_schema_still_accepts_whole_event_sources(event):
    assert_valid_source({"type": "event", "event": event})


def test_schema_still_accepts_any_event_source():
    assert_valid_source({"type": "any_event"})


@pytest.mark.parametrize("kind", [
    "order_book_mid_price", "order_book_spread", "order_book_spread_bps",
    "order_book_weighted_mid_price", "order_book_microprice", "order_book_imbalance",
    "order_book_best_bid_price", "order_book_best_bid_size",
    "order_book_best_ask_price", "order_book_best_ask_size",
])
def test_order_book_definitions_validate_and_round_trip(kind):
    document = document_with_source({"type": "order_book"})
    document["features"][0]["symbol"] = "btcusdt"
    indicator = {"kind": kind, "source": {"type": "order_book"}}
    if kind == "order_book_imbalance":
        indicator["outputs"] = [{"n_levels": 5, "id": "depth_five"}, {"n_levels": 1}]
        document["capacity"] = document["length"] = 2
    document["features"][0]["indicators"] = [indicator]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    assert spec.indicator_count() == 1


@pytest.mark.parametrize("changes", [
    {"outputs": []},
    {"outputs": [{}]},
    {"outputs": [{"n_levels": 0}]},
    {"outputs": [{"n_levels": -1}]},
    {"outputs": [{"n_levels": 1.5}]},
    {"outputs": [{"n_levels": "1"}]},
    {"outputs": [{"n_levels": None}]},
    {"outputs": [{"n_levels": 1, "window": 1}]},
    {"outputs": [{"n_levels": n} for n in range(1, 18)]},
    {"kind": "order_book_mid_price"},
    {"kind": "order_book_mid_price", "outputs": [{}, {}]},
    {"source": {"type": "order_book", "event": "order_book_delta"}},
    {"source": {"type": "order_book", "field": "price"}},
    {"source": {"type": "event", "event": "order_book_snapshot"}},
    {"kind": "sma"},
    {"warmup_policy": "full_window"},
    {"options": {"aggregation": "1s"}},
])
def test_schema_and_reader_reject_invalid_order_book_parameters(changes):
    document = document_with_source({"type": "order_book"})
    indicator = {
        "kind": "order_book_imbalance", "source": {"type": "order_book"},
        "outputs": [{"n_levels": 1}], **changes,
    }
    document["capacity"] = document["length"] = len(indicator["outputs"])
    document["features"][0]["indicators"] = [indicator]
    assert list(VALIDATOR.iter_errors(document))
    with pytest.raises(ValueError):
        fiml.FeatureExtractorSpec.from_json(json.dumps(document))


@pytest.mark.parametrize("kind, parameters", [
    ("level_size", {"price": "100.12345678901234567890123456"}),
    ("nth_price", {"n_levels": 2}),
    ("nth_size", {"n_levels": 2}),
    ("depth_until_price", {"price": "100"}),
    ("depth_until_size_price_from", {"size": "4"}),
    ("depth_until_size_price_to", {"size": "4"}),
    ("depth_until_size_total_size", {"size": "4"}),
    ("volume_between_prices", {"from_price": "99", "to_price": "103"}),
])
@pytest.mark.parametrize("side", ["bid", "ask"])
def test_order_book_query_schema_and_round_trip(kind, parameters, side):
    document = document_with_source({"type": "order_book"})
    document["features"][0]["symbol"] = "btcusdt"
    indicator = {
        "kind": f"order_book_{kind}", "source": {"type": "order_book"},
        "options": {"side": side, **parameters}, "outputs": [{"id": "book_query"}],
    }
    document["features"][0]["indicators"] = [indicator]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    assert spec.indicator_count() == 1
    for invalid in [
        {**indicator, "options": parameters},
        {**indicator, "options": {"side": "wrong", **parameters}},
        {**indicator, "options": {"side": side, **parameters, "aggregation": "1s"}},
        {**indicator, "outputs": [{"n_levels": 1}]},
        {**indicator, "outputs": [{}, {}]},
    ]:
        document["features"][0]["indicators"] = [invalid]
        assert list(VALIDATOR.iter_errors(document))
        with pytest.raises(ValueError):
            fiml.FeatureExtractorSpec.from_json(json.dumps(document))


def test_order_book_configuration_schema_and_canonical_round_trip():
    document = document_with_source({"type": "any_event"})
    document["features"] = []
    document["capacity"] = document["length"] = 0
    document["order_books"] = [
        {"symbol": "eth", "update_policy": "monotonic", "buffer_size": 0},
        {"symbol": "btc", "update_policy": "contiguous", "buffer_size": 8},
    ]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    expected = dict(document, order_books=list(reversed(document["order_books"])))
    assert json.loads(spec.to_json()) == expected
    assert fiml.FeatureExtractor(spec).values().size == 0
    for config in [
        {"symbol": "btc", "update_policy": "bad", "buffer_size": 1},
        {"symbol": "btc", "update_policy": "monotonic", "buffer_size": -1},
        {"symbol": "btc", "update_policy": "monotonic", "buffer_size": 1.5},
        {"symbol": "btc", "update_policy": "monotonic", "buffer_size": None},
        {"symbol": "btc", "buffer_size": 1},
        {"symbol": "", "update_policy": "monotonic", "buffer_size": 1},
        {"symbol": "__global__", "update_policy": "monotonic", "buffer_size": 1},
        {"symbol": "btc", "update_policy": "monotonic", "buffer_size": 1, "bids": []},
    ]:
        invalid = dict(document, order_books=[config])
        assert list(VALIDATOR.iter_errors(invalid))
        with pytest.raises(ValueError):
            fiml.FeatureExtractorSpec.from_json(json.dumps(invalid))
    with pytest.raises(ValueError):
        fiml.FeatureExtractorSpec.from_json(json.dumps(dict(document, order_books=None)))
    duplicate = [dict(document["order_books"][0], symbol=s) for s in ["btc", "BTC"]]
    with pytest.raises(ValueError, match="more than one order book"):
        fiml.FeatureExtractorSpec.from_json(json.dumps(dict(document, order_books=duplicate)))

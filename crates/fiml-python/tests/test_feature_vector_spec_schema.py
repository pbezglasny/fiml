import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).parents[3] / "docs" / "feature-vector-spec.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def document_with_source(source):
    return {
        "version": "1.0",
        "feature_vector_capacity": 1,
        "feature_vector_length": 1,
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


def test_feature_vector_spec_schema_is_valid_draft_2020_12():
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


def test_schema_still_accepts_every_event_source():
    assert_valid_source({"type": "every_event"})

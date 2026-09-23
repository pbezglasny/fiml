import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import numpy as np


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/binance_depth.py"
spec = importlib.util.spec_from_file_location("binance_depth", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_saved_binance_depth_matches_decimal_calculations():
    pipeline = example.build_pipeline()
    actual = pipeline.transform_order_book(example.load_events(example.SAMPLE))
    expected = []
    last_id = -1
    previous_imbalance = np.nan
    for line in example.SAMPLE.read_text().splitlines():
        depth = json.loads(line)["data"]
        if depth["lastUpdateId"] <= last_id:
            continue
        last_id = depth["lastUpdateId"]
        bids = [[Decimal(value) for value in level] for level in depth["bids"]]
        asks = [[Decimal(value) for value in level] for level in depth["asks"]]
        bid, bid_size = bids[0]
        ask, ask_size = asks[0]
        mid = (bid + ask) / 2
        imbalances = []
        for n in [1, 5, 20]:
            bid_volume = sum(size for _, size in bids[:n])
            ask_volume = sum(size for _, size in asks[:n])
            imbalances.append((bid_volume - ask_volume) / (bid_volume + ask_volume))
        expected.append([
            mid, ask - bid, (ask - bid) / mid * 10_000,
            (ask * bid_size + bid * ask_size) / (bid_size + ask_size),
            *imbalances, bid_size, previous_imbalance,
        ])
        previous_imbalance = imbalances[0]
    assert len(expected) > 1
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=1e-12)


def test_partial_snapshots_replace_levels_and_skip_duplicates(tmp_path):
    sample = tmp_path / "depth.jsonl"
    records = [
        {"lastUpdateId": 10, "bids": [["100", "3"]], "asks": [["102", "1"]]},
        {"lastUpdateId": 10, "bids": [["100", "3"]], "asks": [["102", "1"]]},
        {"lastUpdateId": 25, "bids": [["98", "1"]], "asks": [["104", "3"]]},
    ]
    sample.write_text("".join(json.dumps({
        "stream": example.STREAM, "received_at_ms": i, "data": depth,
    }) + "\n" for i, depth in enumerate(records)))
    rows = example.build_pipeline().transform_order_book(example.load_events(sample))
    assert rows.shape == (2, 9)
    np.testing.assert_equal(rows[:, [0, 1, 3, 4, 8]], [
        [101, 2, 101.5, 0.5, np.nan],
        [101, 6, 99.5, -0.5, 0.5],
    ])

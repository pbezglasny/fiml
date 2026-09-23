"""Record Binance Spot partial depth and replay it into model input features.

Run from the repository root:
    uv run --with ./crates/fiml-python python crates/fiml-python/examples/binance_depth.py

Capture a fresh sample (requires websockets; the output file must not exist):
    uv run --with ./crates/fiml-python --with websockets python \
        crates/fiml-python/examples/binance_depth.py --capture 30 --file /tmp/depth.jsonl

Protocol: https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#partial-book-depth-streams
Each message replaces the top 20 levels. This is not the diff-depth stream.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import fiml
import numpy as np


SYMBOL = "BTCUSDT"
STREAM = "wss://data-stream.binance.vision/ws/btcusdt@depth20@100ms"
SAMPLE = Path(__file__).with_name("binance_depth.jsonl")


def capture(path: Path, count: int) -> None:
    """Save untouched depth payloads with their source and local receive time."""
    from websockets.sync.client import connect

    if count <= 0:
        raise ValueError("capture count must be positive")
    with path.open("x", encoding="utf-8") as output:
        with connect(STREAM, open_timeout=15) as websocket:
            for _ in range(count):
                payload = websocket.recv(timeout=15)
                # Partial depth has no exchange timestamp; use receive epoch ms.
                record = {
                    "stream": STREAM,
                    "received_at_ms": time.time_ns() // 1_000_000,
                    "data": json.loads(payload),
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_events(path: Path) -> list:
    """Convert newer partial-depth messages to exact-decimal FIML snapshots."""
    events = []
    last_update_id = -1
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["stream"] != STREAM:
                raise ValueError(f"expected {STREAM}")
            depth = record["data"]
            update_id = depth["lastUpdateId"]
            # An unchanged book can be sent again; don't advance lag history.
            if update_id <= last_update_id:
                continue
            events.append(fiml.OrderBookEvent.snapshot(
                SYMBOL, record["received_at_ms"], update_id,
                depth["bids"], depth["asks"],
            ))
            last_update_id = update_id
    if not events:
        raise ValueError("sample contains no depth snapshots")
    return events


def build_pipeline() -> fiml.ModelInputPipeline:
    """Build named model inputs, including imbalance from the previous snapshot."""
    # Snapshot IDs can jump; no delta history needs to be retained.
    # ponytail: only the top 20 levels; use REST + diff depth for deeper queries.
    raw = (fiml.FeatureExtractorSpec()
           .configure_order_book(SYMBOL, update_policy="monotonic", buffer_size=0)
           .order_book_mid_price(SYMBOL)
           .order_book_spread(SYMBOL)
           .order_book_spread_bps(SYMBOL)
           .order_book_microprice(SYMBOL)
           .order_book_imbalance(SYMBOL, [1, 5, 20])
           .order_book_best_bid_size(SYMBOL))
    names = [
        "mid_price", "spread", "spread_bps", "microprice",
        "imbalance_1", "imbalance_5", "imbalance_20", "best_bid_size",
    ]
    spec = fiml.PipelineSpec(raw)
    for feature_id, name in zip(raw.feature_ids(), names, strict=True):
        spec.identity(feature_id, output=name)
    spec.lagged(raw.feature_ids()[4], lag_window=1, output="previous_imbalance_1")
    # The serialized recipe can also be loaded by Rust.
    return fiml.ModelInputPipeline.from_json(spec.to_json())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=SAMPLE)
    parser.add_argument("--capture", type=int, metavar="COUNT")
    args = parser.parse_args()
    if args.capture is not None:
        capture(args.file, args.capture)

    events = load_events(args.file)
    pipeline = build_pipeline()
    features = pipeline.transform_order_book(events)
    # One row per newer snapshot, ready for an ML model. The first lag is NaN.
    np.savetxt(sys.stdout, features, delimiter=",", fmt="%.10g",
               header=",".join(pipeline.feature_names()), comments="")


if __name__ == "__main__":
    main()

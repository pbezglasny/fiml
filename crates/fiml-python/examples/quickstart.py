"""Quickstart + parity check for the fiml Python bindings.

Run after ``maturin develop``:

    python examples/quickstart.py

It builds an engine from an EngineSpec JSON, computes features on a small price
series with ``transform`` (the batch path), and verifies that the batch result
exactly equals stepping the same events one at a time with ``update`` (the
streaming path). Both go through the same Rust dispatch, so they must match
exactly. It then shows the keyword payload columns each event kind uses.
"""

import json

import numpy as np

import fiml

SPEC = {
    "features": [
        {"name": "sma_3", "symbol": "BTCUSDT", "spec": {"Sma": {"period": 3}}},
        {"name": "ema_3", "symbol": "BTCUSDT", "spec": {"Ema": {"period": 3}}},
    ]
}


def main() -> None:
    engine = fiml.Engine.from_spec_json(json.dumps(SPEC))
    btc = engine.symbol("BTCUSDT")

    prices = np.array([10.0, 11.0, 9.0, 12.0, 13.0, 12.5], dtype=np.float64)
    n = prices.shape[0]
    kind = np.full(n, fiml.KIND_PRICE, dtype=np.uint8)
    symbol = np.full(n, btc, dtype=np.int64)
    timestamp = np.arange(n, dtype=np.int64)

    # Batch path. Payload columns are keyword-only; KIND_PRICE reads `price`.
    batch = engine.transform(kind, symbol, timestamp, price=prices)

    # Streaming path: a fresh engine stepped one event at a time.
    streaming_engine = fiml.Engine.from_spec_json(json.dumps(SPEC))
    streaming_engine.symbol("BTCUSDT")
    streaming = np.empty_like(batch)
    for i in range(n):
        streaming_engine.update(
            int(kind[i]), int(symbol[i]), int(timestamp[i]), price=float(prices[i])
        )
        streaming[i] = streaming_engine.values()

    columns = engine.feature_names()
    print("columns:", columns)
    print(batch)

    # Exact equality, not approximate: identical spec + identical code path.
    assert np.array_equal(batch, streaming), "batch and streaming outputs diverged"
    print("OK: batch == streaming (exact)")

    check_event_kinds()


def check_event_kinds() -> None:
    """Each event kind reads only the keyword columns it needs.

    KIND_TRADE uses ``price`` + ``volume``; KIND_ORDERBOOK uses ``bid`` + ``ask``
    (it dispatches fine even though no builtin feature subscribes to it yet). A
    kind whose required column is missing raises ``ValueError``.
    """
    engine = fiml.Engine.from_spec_json(json.dumps(SPEC))
    btc = engine.symbol("BTCUSDT")

    kind = np.array([fiml.KIND_TRADE, fiml.KIND_ORDERBOOK], dtype=np.uint8)
    symbol = np.full(2, btc, dtype=np.int64)
    timestamp = np.arange(2, dtype=np.int64)
    price = np.array([10.0, 0.0], dtype=np.float64)
    volume = np.array([1.0, 0.0], dtype=np.float64)
    bid = np.array([0.0, 9.5], dtype=np.float64)
    ask = np.array([0.0, 10.5], dtype=np.float64)

    out = engine.transform(
        kind, symbol, timestamp, price=price, volume=volume, bid=bid, ask=ask
    )
    assert out.shape == (2, engine.n_features())
    print("OK: trade + orderbook rows dispatched")

    # KIND_TRADE needs a `price` column; omitting it is a ValueError.
    try:
        engine.transform(kind[:1], symbol[:1], timestamp[:1])
    except ValueError as err:
        print(f"OK: missing column raised ValueError: {err}")
    else:
        raise AssertionError("expected ValueError for missing `price` column")


if __name__ == "__main__":
    main()

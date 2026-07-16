"""Quickstart + parity check for the fiml Python bindings.

Run after ``maturin develop``:

    python examples/quickstart.py

It authors a ``FeatureSet`` with the fluent builder, computes features on a
small trade frame with ``compute_features`` (the DataFrame path), verifies the
low-level batch ``transform`` exactly equals stepping the same events one at a
time with ``update`` (the streaming path), and shows the keyword payload
columns each event kind uses. Batch and streaming go through the same Rust
dispatch, so they must match exactly.
"""

import numpy as np

import fiml

# 2021-01-01 00:00:00 UTC in epoch milliseconds; events are 1s apart.
T0 = 1_609_459_200_000


def build_feature_set() -> fiml.FeatureSet:
    return (
        fiml.FeatureSet()
        .sma("BTCUSDT", [3])
        .ema("BTCUSDT", [3])
        .day_of_week()
    )


def main() -> None:
    fs = build_feature_set()
    extractor = fiml.FeatureExtractor(fs)
    btc = extractor.symbol("BTCUSDT")

    prices = np.array([10.0, 11.0, 9.0, 12.0, 13.0, 12.5], dtype=np.float64)
    n = prices.shape[0]
    kind = np.full(n, fiml.KIND_PRICE, dtype=np.uint8)
    symbol = np.full(n, btc, dtype=np.int64)
    timestamp = T0 + np.arange(n, dtype=np.int64) * 1_000

    # Batch path. Payload columns are keyword-only; KIND_PRICE reads `price`.
    batch = extractor.transform(kind, symbol, timestamp, price=prices)

    # Streaming path: a fresh extractor stepped one event at a time. Built from
    # the FeatureSet JSON round-trip — the parity artifact saved next to a
    # trained model and loaded by Rust serving.
    streaming_extractor = fiml.FeatureExtractor.from_json(fs.to_json())
    streaming_extractor.symbol("BTCUSDT")
    streaming = np.empty_like(batch)
    for i in range(n):
        streaming_extractor.update(
            int(kind[i]), int(symbol[i]), int(timestamp[i]), price=float(prices[i])
        )
        streaming[i] = streaming_extractor.values()

    print("columns:", extractor.feature_names())
    print(batch)

    # Exact equality, not approximate: identical feature set + identical code
    # path. equal_nan covers warmup cells, which read NaN until first write.
    assert np.array_equal(batch, streaming, equal_nan=True), (
        "batch and streaming outputs diverged"
    )
    print("OK: batch == streaming (exact)")

    check_compute_features()
    check_event_kinds()


def check_compute_features() -> None:
    """The trade DataFrame path: one snapshot per row, including metadata."""
    try:
        import pandas as pd
    except ImportError:
        print("skipped compute_features demo (pandas not installed)")
        return

    df = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "ts": T0 + np.arange(6, dtype=np.int64) * 1_000,
            "price": [10.0, 11.0, 9.0, 12.0, 13.0, 12.5],
            "volume": [1.0, 2.0, 1.5, 3.0, 2.5, 2.0],
        }
    )

    feature_set = (
        fiml.FeatureSet()
        .sma("BTCUSDT", [3], source="trade_price")
        .ema("BTCUSDT", [3], source="trade_price")
        .obv_timed("BTCUSDT", aggregation="1s", windows=["60s"])
        .trade_count_timed("BTCUSDT", aggregation="1s", window="60s")
        .day_of_week()
    )
    extractor = fiml.FeatureExtractor(feature_set, output_dtype=np.float32)
    feats = extractor.compute_features(df)
    assert feats.shape == (len(df), extractor.n_features() + 2)
    assert list(feats.columns) == ["symbol", "ts", *extractor.feature_names()]
    assert (feats.index == df.index).all()
    assert all(dtype == np.float32 for dtype in feats.dtypes.iloc[2:])
    print("OK: compute_features returned one aligned snapshot per trade")
    print(feats)


def check_event_kinds() -> None:
    """Each event kind reads only the keyword columns it needs.

    KIND_TRADE uses ``price`` + ``volume``; KIND_ORDERBOOK uses ``bid`` + ``ask``
    (it dispatches fine even though no builtin feature subscribes to it yet). A
    kind whose required column is missing raises ``ValueError`` without
    mutating extractor state (rows are validated before any dispatch).
    """
    extractor = fiml.FeatureExtractor(
        fiml.FeatureSet().trade_count_timed("BTCUSDT", aggregation="1s", window="60s")
    )
    btc = extractor.symbol("BTCUSDT")

    kind = np.array([fiml.KIND_TRADE, fiml.KIND_ORDERBOOK], dtype=np.uint8)
    symbol = np.full(2, btc, dtype=np.int64)
    timestamp = T0 + np.arange(2, dtype=np.int64) * 1_000
    price = np.array([10.0, 0.0], dtype=np.float64)
    volume = np.array([1.0, 0.0], dtype=np.float64)
    bid = np.array([0.0, 9.5], dtype=np.float64)
    ask = np.array([0.0, 10.5], dtype=np.float64)

    out = extractor.transform(
        kind, symbol, timestamp, price=price, volume=volume, bid=bid, ask=ask
    )
    assert out.shape == (2, extractor.n_features())
    assert out[0, 0] == 1.0 and out[1, 0] == 1.0, "trade count should see one trade"
    print("OK: trade + orderbook rows dispatched")

    # KIND_TRADE needs a `price` column; omitting it is a ValueError.
    try:
        extractor.transform(kind[:1], symbol[:1], timestamp[:1])
    except ValueError as err:
        print(f"OK: missing column raised ValueError: {err}")
    else:
        raise AssertionError("expected ValueError for missing `price` column")


if __name__ == "__main__":
    main()

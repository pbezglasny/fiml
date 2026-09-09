"""Load and normalize the Binance trade sample shared by the notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd

import fiml


def load_binance_trades() -> pd.DataFrame:
    """Return timestamp-ordered trades with FIML volume and aggressor-side columns."""
    trades = (
        pd.read_csv(
            Path(__file__).with_name("price_data_binance_trades.csv"),
            dtype={
                "symbol": "string",
                "trade_id": "int64",
                "price": "float64",
                "quantity": "float64",
                "buyer_market_maker": "boolean",
            },
            parse_dates=["time", "trade_time"],
        )
        .rename(columns={"quantity": "volume"})
        .sort_values(["trade_time", "trade_id"], kind="stable")
        .reset_index(drop=True)
    )
    trades["ts"] = trades["trade_time"].dt.as_unit("ms").astype("int64")
    trades["side"] = np.where(
        trades["buyer_market_maker"].to_numpy(dtype=bool),
        fiml.SIDE_AGGRESSOR_SELL,
        fiml.SIDE_AGGRESSOR_BUY,
    ).astype(np.uint8)

    assert trades["ts"].is_monotonic_increasing
    assert (
        trades.loc[trades["buyer_market_maker"], "side"]
        == fiml.SIDE_AGGRESSOR_SELL
    ).all()
    assert (
        trades.loc[~trades["buyer_market_maker"], "side"]
        == fiml.SIDE_AGGRESSOR_BUY
    ).all()
    return trades

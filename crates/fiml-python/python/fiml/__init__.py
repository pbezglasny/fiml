"""Python bindings for the fiml indicator engine.

Features are computed by the exact Rust engine (the same code the live Rust
environment runs), so batch (training) and live (serving) outputs match given the
same spec and the same event stream. See the package README for the determinism
rules.
"""

from ._fiml import (
    Engine,
    KIND_PRICE,
    KIND_VOLUME,
    KIND_TRADE,
    KIND_ORDERBOOK,
    KIND_TIME,
)

__all__ = [
    "Engine",
    "KIND_PRICE",
    "KIND_VOLUME",
    "KIND_TRADE",
    "KIND_ORDERBOOK",
    "KIND_TIME",
]

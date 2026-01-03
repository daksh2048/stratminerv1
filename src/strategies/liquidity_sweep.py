from __future__ import annotations

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _to_float(x):
    """
    Robustly convert a pandas scalar / Series / numpy scalar to a Python float.
    Avoids calling float() directly on a Series, which triggers FutureWarnings.
    """
    import pandas as _pd
    import numpy as _np

    # If it's a Series / Index, take the first element
    if isinstance(x, (_pd.Series, _pd.Index)):
        if len(x) == 0:
            return float("nan")
        return float(x.iloc[0])

    # If it's a numpy scalar, unwrap it
    if isinstance(x, _np.generic):
        return float(x)

    # Otherwise, let float() handle it
    return float(x)


class LiquiditySweep(Strategy):
    """
    Simple liquidity sweep strategy:

    - Bearish setup:
        * Last candle's HIGH takes out the recent high of the lookback window
        * Candle closes BEARISH (close < open)
        -> Sell at close, stop at high, TP at RR * (stop - entry)

    - Bullish setup:
        * Last candle's LOW takes out the recent low of the lookback window
        * Candle closes BULLISH (close > open)
        -> Buy at close, stop at low, TP at RR * (entry - stop)
    """

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        lb = int(self.params.get("lookback", 10))
        rr = float(self.params.get("risk_reward", 1.5))

        # Need enough candles to form a window + last candle
        if len(df) < lb + 2:
            return Order(symbol, None, None, None, None, "Not enough data", {})

        # ----- Last candle as plain floats -----
        last = df.iloc[-1]
        last_high = _to_float(last["high"])
        last_low = _to_float(last["low"])
        last_open = _to_float(last["open"])
        last_close = _to_float(last["close"])

        # ----- Recent highs/lows from the preceding window -----
        window = df.iloc[-(lb + 1):-1]
        recent_high = _to_float(window["high"].max())
        recent_low = _to_float(window["low"].min())

        # ----- Bearish liquidity sweep -----
        # Wick above recent highs, but candle closes down (rejection)
        if (last_high > recent_high) and (last_close < last_open):
            entry = last_close
            stop = last_high
            take = entry - rr * (stop - entry)
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="bearish liquidity sweep",
                meta={"recent_high": recent_high},
            )

        # ----- Bullish liquidity sweep -----
        # Wick below recent lows, but candle closes up (rejection)
        if (last_low < recent_low) and (last_close > last_open):
            entry = last_close
            stop = last_low
            take = entry + rr * (entry - stop)
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="bullish liquidity sweep",
                meta={"recent_low": recent_low},
            )

        # ----- No setup -----
        return Order(symbol, None, None, None, None, "no setup", {})

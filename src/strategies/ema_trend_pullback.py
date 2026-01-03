from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _to_float(x: Any) -> float:
    """
    Robustly convert a pandas / numpy scalar or Series to a Python float.
    Mirrors the helper used in other strategies to avoid weird type issues.
    """
    import numpy as _np
    import pandas as _pd

    if isinstance(x, (float, int)):
        return float(x)

    # pandas / numpy scalar
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass

    # Series / Index → take last element
    if isinstance(x, (_pd.Series, _pd.Index)):
        return float(x.iloc[-1])

    return float(x)


class EMATrendPullback(Strategy):
    """
    Basic EMA trend + pullback strategy.

    Idea:
    - Compute fast and slow EMAs on close.
    - Uptrend  = fast EMA > slow EMA.
      Downtrend = fast EMA < slow EMA.
    - Wait for price to pull back close to the fast EMA and print
      a candle in the direction of the trend.
    - Use a recent swing high/low for stop.
    - Take-profit is a fixed risk:reward multiple.

    Parameters (with defaults):
    - fast_ema        (int)   : default 9
    - slow_ema        (int)   : default 21
    - swing_lookback  (int)   : default 3  (for stop)
    - max_distance_pct(float) : default 0.0025 (0.25% from fast EMA)
    - risk_reward     (float) : default 2.0
    """

    def __init__(self, name: str, **params) -> None:
        super().__init__(name, **params)

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # ---- Read params with sane defaults ----
        fast = int(self.params.get("fast_ema", 9))
        slow = int(self.params.get("slow_ema", 21))
        swing_lookback = int(self.params.get("swing_lookback", 3))
        max_distance_pct = float(self.params.get("max_distance_pct", 0.0025))
        rr = float(self.params.get("risk_reward", 2.0))

        # Need enough history for EMA calculations
        if len(df) < slow + 5:
            return Order(symbol, None, None, None, None, "not enough data", {})

        closes = df["close"]

        # ---- Indicator calculations ----
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()

        last_close = _to_float(closes.iloc[-1])
        last_open = _to_float(df["open"].iloc[-1])
        last_high = _to_float(df["high"].iloc[-1])
        last_low = _to_float(df["low"].iloc[-1])

        fast_val = _to_float(ema_fast.iloc[-1])
        slow_val = _to_float(ema_slow.iloc[-1])

        # Distance of price from fast EMA
        distance_pct = abs(last_close - fast_val) / fast_val

        # ---- Uptrend: fast EMA above slow EMA ----
        if (fast_val > slow_val) and (distance_pct <= max_distance_pct) and (last_close > last_open):
            recent_low = _to_float(df["low"].iloc[-swing_lookback:].min())

            # Safety check: stop must be below entry
            if recent_low >= last_close:
                return Order(symbol, None, None, None, None, "invalid long stop", {})

            entry = last_close
            stop = recent_low
            take = entry + rr * (entry - stop)

            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="EMA pullback long",
                meta={
                    "ema_fast": fast_val,
                    "ema_slow": slow_val,
                    "distance_pct": distance_pct,
                },
            )

        # ---- Downtrend: fast EMA below slow EMA ----
        if (fast_val < slow_val) and (distance_pct <= max_distance_pct) and (last_close < last_open):
            recent_high = _to_float(df["high"].iloc[-swing_lookback:].max())

            # Safety check: stop must be above entry
            if recent_high <= last_close:
                return Order(symbol, None, None, None, None, "invalid short stop", {})

            entry = last_close
            stop = recent_high
            take = entry - rr * (stop - entry)

            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="EMA pullback short",
                meta={
                    "ema_fast": fast_val,
                    "ema_slow": slow_val,
                    "distance_pct": distance_pct,
                },
            )

        # ---- No setup on this candle ----
        return Order(symbol, None, None, None, None, "no EMA pullback setup", {})


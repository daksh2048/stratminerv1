from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _as_float(x) -> float:
    # Future-proof float conversion when x is a 1-element Series
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def _to_market_tz(ts: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    # If ts is naive, assume UTC (consistent default)
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(market_tz)


@dataclass
class ORBState:
    last_trade_day: Optional[pd.Timestamp] = None  # date normalized in market tz


class OpeningRangeBreakout(Strategy):
    """
    Opening Range Breakout (ORB) with Phase-1 sanity gates:
      - RTH only (09:30–16:00 ET)
      - Entry cutoff (minutes after open, e.g. 60 => until 10:30 ET)
      - Once per day (per symbol)

    NOTE: This matches your project's Order signature by ALWAYS providing `reason` and `meta`.
    """

    def __init__(self, name: str = "orb", **params) -> None:
        super().__init__(name, **params)
        self._state_by_symbol: dict[str, ORBState] = {}

    def _get_state(self, symbol: str) -> ORBState:
        if symbol not in self._state_by_symbol:
            self._state_by_symbol[symbol] = ORBState()
        return self._state_by_symbol[symbol]

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params (defaults) ---
        or_bars = int(self.params.get("or_bars", 6))
        rr = float(self.params.get("risk_reward", 2.0))
        once_per_day = bool(self.params.get("once_per_day", True))

        # Phase-1 gates
        rth_only = bool(self.params.get("rth_only", True))
        entry_cutoff_minutes = self.params.get("entry_cutoff_minutes", 60)  # None disables
        entry_cutoff_minutes = None if entry_cutoff_minutes is None else int(entry_cutoff_minutes)

        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open_str = str(self.params.get("rth_open", "09:30"))
        rth_close_str = str(self.params.get("rth_close", "16:00"))

        if df is None or len(df) < (or_bars + 5):
            return Order(symbol, None, None, None, None, "not enough candles", {})

        # --- last timestamp in market tz ---
        last_ts = pd.Timestamp(df.index[-1])
        last_ts_m = _to_market_tz(last_ts, market_tz)
        day_m = last_ts_m.normalize()

        # --- session bounds for that day ---
        open_h, open_m = map(int, rth_open_str.split(":"))
        close_h, close_m = map(int, rth_close_str.split(":"))

        session_open = day_m + pd.Timedelta(hours=open_h, minutes=open_m)
        session_close = day_m + pd.Timedelta(hours=close_h, minutes=close_m)

        # RTH-only gate
        if rth_only and not (session_open <= last_ts_m <= session_close):
            return Order(symbol, None, None, None, None, "outside RTH", {"ts": str(last_ts_m)})

        # entry cutoff gate
        if entry_cutoff_minutes is not None:
            cutoff_ts = session_open + pd.Timedelta(minutes=entry_cutoff_minutes)
            if last_ts_m > cutoff_ts:
                return Order(
                    symbol, None, None, None, None,
                    "past entry cutoff",
                    {"ts": str(last_ts_m), "cutoff": str(cutoff_ts)},
                )

        # --- build today's RTH candles only ---
        idx = pd.to_datetime(df.index)

        if getattr(idx, "tz", None) is None:
            idx_m = idx.tz_localize("UTC").tz_convert(market_tz)
        else:
            idx_m = idx.tz_convert(market_tz)

        in_today = (idx_m.normalize() == day_m)
        in_session = (idx_m >= session_open) & (idx_m <= session_close)
        day_rth = df.loc[in_today & in_session]

        if len(day_rth) < or_bars + 1:
            return Order(symbol, None, None, None, None, "waiting for OR to form", {})

        opening_segment = day_rth.iloc[:or_bars]
        last = day_rth.iloc[-1]

        or_high = _as_float(opening_segment["high"].max())
        or_low = _as_float(opening_segment["low"].min())
        last_open = _as_float(last["open"])
        last_close = _as_float(last["close"])

        # once-per-day gate (per symbol)
        st = self._get_state(symbol)
        if once_per_day and st.last_trade_day is not None and st.last_trade_day == day_m:
            return Order(symbol, None, None, None, None, "once_per_day already traded", {"day": str(day_m)})

        # --- LONG breakout ---
        if last_close > or_high and last_open <= or_high:
            entry = last_close
            stop = or_low
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop (OR too tight)", {"or_high": or_high, "or_low": or_low})

            take = entry + rr * (entry - stop)
            st.last_trade_day = day_m

            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="ORB breakout long",
                meta={
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_bars": or_bars,
                    "market_tz": market_tz,
                    "session_open": str(session_open),
                    "session_close": str(session_close),
                    "entry_cutoff_minutes": entry_cutoff_minutes,

                    "trail_mode": str(self.params.get("trail_mode", "percent")).lower(),
                    "trail_pct": self.params.get("trail_pct", None),              # only used for percent mode
                    "trail_atr_mult": self.params.get("trail_atr_mult", 2.0),
                    "trail_lookback": self.params.get("trail_lookback", 10),
                    "trail_activate_after_partial": self.params.get("trail_activate_after_partial", False),
                },
            )

        # --- SHORT breakout ---
        if last_close < or_low and last_open >= or_low:
            entry = last_close
            stop = or_high
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop (OR too tight)", {"or_high": or_high, "or_low": or_low})

            take = entry - rr * (stop - entry)
            st.last_trade_day = day_m

            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="ORB breakout short",
                meta={
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_bars": or_bars,
                    "market_tz": market_tz,
                    "session_open": str(session_open),
                    "session_close": str(session_close),
                    "entry_cutoff_minutes": entry_cutoff_minutes,
                },
            )

        return Order(symbol, None, None, None, None, "no ORB setup", {})

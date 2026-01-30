from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _as_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def _to_market_tz(ts: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(market_tz)


@dataclass
class VWAPState:
    day: Optional[pd.Timestamp] = None        # normalized day in market tz
    trades_today: int = 0


class VWAPTrendPullback(Strategy):
    """
    VWAP Trend Pullback (intraday)

    Bias:
      - Long bias when close > VWAP and close > EMA
      - Short bias when close < VWAP and close < EMA

    Entry (reclaim after touch):
      - Long: candle touches/pierces VWAP (low <= vwap) AND closes back above VWAP
              AND prior candle was below/near VWAP (pullback context)
      - Short: candle touches/pierces VWAP (high >= vwap) AND closes back below VWAP
              AND prior candle was above/near VWAP

    Risk:
      - Stop uses ATR buffer beyond min(low, vwap) / max(high, vwap)
      - Take = RR * risk

    Phase-1 gates:
      - RTH only
      - Entry cutoff minutes from open
      - max_trades_per_day (per symbol)
    """

    def __init__(self, name: str = "vwap_tp", **params) -> None:
        super().__init__(name, **params)
        self._state_by_symbol: dict[str, VWAPState] = {}

    def _get_state(self, symbol: str) -> VWAPState:
        if symbol not in self._state_by_symbol:
            self._state_by_symbol[symbol] = VWAPState()
        return self._state_by_symbol[symbol]

    def _session_bounds(self, day_m: pd.Timestamp, market_tz: str, rth_open: str, rth_close: str):
        open_h, open_m = map(int, rth_open.split(":"))
        close_h, close_m = map(int, rth_close.split(":"))

        session_open = day_m.normalize() + pd.Timedelta(hours=open_h, minutes=open_m)
        session_close = day_m.normalize() + pd.Timedelta(hours=close_h, minutes=close_m)
        # already in market tz
        return session_open, session_close

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # ---- params (defaults) ----
        ema_period = int(self.params.get("ema_period", 20))
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_mult", 1.0))
        rr = float(self.params.get("risk_reward", 2.0))

        rth_only = bool(self.params.get("rth_only", True))
        entry_cutoff_minutes = self.params.get("entry_cutoff_minutes", 240)  # default: allow first 4 hours
        entry_cutoff_minutes = None if entry_cutoff_minutes is None else int(entry_cutoff_minutes)

        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))

        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))

        if df is None or len(df) < max(ema_period, atr_period) + 5:
            return Order(symbol, None, None, None, None, "not enough candles", {})

        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume for VWAP", {})

        # ---- last timestamp in market tz ----
        last_ts = pd.Timestamp(df.index[-1])
        last_ts_m = _to_market_tz(last_ts, market_tz)
        day_m = last_ts_m.normalize()

        session_open, session_close = self._session_bounds(day_m, market_tz, rth_open, rth_close)

        # RTH-only gate
        if rth_only and not (session_open <= last_ts_m <= session_close):
            return Order(symbol, None, None, None, None, "outside RTH", {"ts": str(last_ts_m)})

        # Entry cutoff gate
        if entry_cutoff_minutes is not None:
            cutoff_ts = session_open + pd.Timedelta(minutes=entry_cutoff_minutes)
            if last_ts_m > cutoff_ts:
                return Order(symbol, None, None, None, None, "past entry cutoff", {"ts": str(last_ts_m), "cutoff": str(cutoff_ts)})

        # ---- Build today's RTH slice ----
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is None:
            idx_m = idx.tz_localize("UTC").tz_convert(market_tz)
        else:
            idx_m = idx.tz_convert(market_tz)

        in_today = (idx_m.normalize() == day_m)
        in_session = (idx_m >= session_open) & (idx_m <= session_close)
        day_rth = df.loc[in_today & in_session].copy()

        if len(day_rth) < max(ema_period, atr_period) + 2:
            return Order(symbol, None, None, None, None, "waiting for indicators warmup", {})

        # ---- per-symbol day state reset ----
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades per day reached", {"day": str(day_m)})

        # ---- VWAP (session) ----
        vol = day_rth["volume"].astype(float)
        vol_sum = float(vol.to_numpy().sum())
        if vol_sum <= 0:
            return Order(symbol, None, None, None, None, "zero volume (cannot compute VWAP)", {})

        typical = (day_rth["high"].astype(float) + day_rth["low"].astype(float) + day_rth["close"].astype(float)) / 3.0
        cum_pv = (typical * vol).cumsum()
        cum_v = vol.cumsum()
        vwap = cum_pv / cum_v

        # ---- EMA ----
        close = day_rth["close"].astype(float)
        ema = close.ewm(span=ema_period, adjust=False).mean()

        # ---- ATR ----
        high = day_rth["high"].astype(float)
        low = day_rth["low"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(atr_period).mean()

        # Need last + prev for pullback logic
        if len(day_rth) < 3:
            return Order(symbol, None, None, None, None, "not enough RTH candles", {})

        last = day_rth.iloc[-1]
        prev = day_rth.iloc[-2]

        vwap_now = _as_float(vwap.iloc[-1])
        vwap_prev = _as_float(vwap.iloc[-2])
        ema_now = _as_float(ema.iloc[-1])
        atr_now = _as_float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None

        if atr_now is None or atr_now <= 0:
            return Order(symbol, None, None, None, None, "ATR not ready", {})

        last_close = _as_float(last["close"])
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])

        prev_close_f = _as_float(prev["close"])
        prev_high_f = _as_float(prev["high"])
        prev_low_f = _as_float(prev["low"])

        # ---- Bias ----
        bias_long = (last_close > vwap_now) and (last_close > ema_now)
        bias_short = (last_close < vwap_now) and (last_close < ema_now)

        # ---- Pullback + reclaim conditions ----
        long_reclaim = (last_low <= vwap_now) and (last_close > vwap_now)
        long_pullback_context = (prev_close_f <= vwap_prev) or (prev_low_f <= vwap_prev)

        short_reclaim = (last_high >= vwap_now) and (last_close < vwap_now)
        short_pullback_context = (prev_close_f >= vwap_prev) or (prev_high_f >= vwap_prev)

        # ---- LONG entry ----
        if bias_long and long_pullback_context and long_reclaim:
            entry = last_close
            stop = min(last_low, vwap_now) - atr_mult * atr_now
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop", {"vwap": vwap_now, "atr": atr_now})

            take = entry + rr * (entry - stop)
            st.trades_today += 1

            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="VWAP pullback long (touch + reclaim in trend)",
                meta={
                    "vwap": vwap_now,
                    "ema": ema_now,
                    "atr": atr_now,
                    "ema_period": ema_period,
                    "atr_period": atr_period,
                    "atr_mult": atr_mult,
                    "risk_reward": rr,
                    "session_open": str(session_open),
                    "entry_cutoff_minutes": entry_cutoff_minutes,
                    "trades_today": st.trades_today,
                },
            )

        # ---- SHORT entry ----
        if bias_short and short_pullback_context and short_reclaim:
            entry = last_close
            stop = max(last_high, vwap_now) + atr_mult * atr_now
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop", {"vwap": vwap_now, "atr": atr_now})

            take = entry - rr * (stop - entry)
            st.trades_today += 1

            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="VWAP pullback short (touch + reclaim in trend)",
                meta={
                    "vwap": vwap_now,
                    "ema": ema_now,
                    "atr": atr_now,
                    "ema_period": ema_period,
                    "atr_period": atr_period,
                    "atr_mult": atr_mult,
                    "risk_reward": rr,
                    "session_open": str(session_open),
                    "entry_cutoff_minutes": entry_cutoff_minutes,
                    "trades_today": st.trades_today,
                },
            )

        return Order(symbol, None, None, None, None, "no VWAP pullback setup", {"vwap": vwap_now, "ema": ema_now})

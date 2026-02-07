from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _as_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def _index_to_market_tz(index: pd.DatetimeIndex, market_tz: str) -> pd.DatetimeIndex:
    """
    Match the logic already used in your current VWAP MR:
      - naive -> assume already market_tz
      - aware -> convert
    """
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize(market_tz)
    return idx.tz_convert(market_tz)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _session_bounds(day_m: pd.Timestamp, rth_open: str, rth_close: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    oh, om = map(int, rth_open.split(":"))
    ch, cm = map(int, rth_close.split(":"))
    session_open = day_m.normalize() + pd.Timedelta(hours=oh, minutes=om)
    session_close = day_m.normalize() + pd.Timedelta(hours=ch, minutes=cm)
    return session_open, session_close


def _vwap_intraday(day_df: pd.DataFrame) -> pd.Series:
    # Typical price VWAP
    tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
    pv = tp * day_df["volume"]
    v = day_df["volume"].replace(0, pd.NA)
    return pv.cumsum() / v.cumsum()


@dataclass
class VWAPMRState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class VWAPMeanReversion(Strategy):
    """
    VWAP Mean Reversion v2 (drop-in)

    Goal: stop catching falling knives.
    Default behavior:
      - Only trade in *non-trending* regimes (EMA slope small)
      - Only enter after "stretch -> re-entry" (NOT just a 1-bar reversal)
      - Limit to 1 trade/day/symbol by default
      - Avoid first N minutes after open
    """

    def __init__(self, name: str = "vwap_mr", **params) -> None:
        super().__init__(name, **params)
        self._state_by_symbol: dict[str, VWAPMRState] = {}

    def _get_state(self, symbol: str) -> VWAPMRState:
        if symbol not in self._state_by_symbol:
            self._state_by_symbol[symbol] = VWAPMRState()
        return self._state_by_symbol[symbol]

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # ---- params ----
        atr_period = int(self.params.get("atr_period", 14))
        ema_period = int(self.params.get("ema_period", 50))

        # MR trigger bands
        stretch_atr_mult = float(self.params.get("stretch_atr_mult", 1.4))   # must stretch THIS far first
        reentry_atr_mult = float(self.params.get("reentry_atr_mult", 0.6))   # then re-enter to THIS band

        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.0))

        # regime filter: "range only"
        use_regime_filter = bool(self.params.get("use_regime_filter", True))
        ema_slope_max_atr = float(self.params.get("ema_slope_max_atr", 0.18))

        # session
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_only = bool(self.params.get("rth_only", True))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 15))
        entry_cutoff_minutes = self.params.get("entry_cutoff_minutes", 240)
        entry_cutoff_minutes = None if entry_cutoff_minutes is None else int(entry_cutoff_minutes)

        max_trades_per_day = int(self.params.get("max_trades_per_day", 1))

        # per-trade hard cap (optional; broker is account-level)
        max_stop_pct = float(self.params.get("max_stop_pct", 0.03))

        warmup = max(atr_period + 20, ema_period + 5)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "not enough candles", {})

        # ---- build today's session slice ----
        idx_m = _index_to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()

        if rth_only:
            session_open, session_close = _session_bounds(day_m, rth_open, rth_close)
            in_today = (idx_m.normalize() == day_m)
            in_session = (idx_m >= session_open) & (idx_m <= session_close)
            day_df = df.loc[in_today & in_session]
            if day_df.empty:
                return Order(symbol, None, None, None, None, "no RTH candles", {})
        else:
            day_df = df.copy()
            session_open = day_m.normalize()
            session_close = day_m.normalize() + pd.Timedelta(hours=23, minutes=59)

        # wait after open
        if min_minutes_after_open > 0:
            if last_ts_m < (session_open + pd.Timedelta(minutes=min_minutes_after_open)):
                return Order(symbol, None, None, None, None, "too early after open", {"ts": str(last_ts_m)})

        # cutoff
        if entry_cutoff_minutes is not None:
            cutoff_ts = session_open + pd.Timedelta(minutes=entry_cutoff_minutes)
            if last_ts_m > cutoff_ts:
                return Order(symbol, None, None, None, None, "past entry cutoff", {"ts": str(last_ts_m)})

        # day state / trade limit
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day reached", {"day": str(day_m)})

        if len(day_df) < max(ema_period, atr_period) + 3:
            return Order(symbol, None, None, None, None, "not enough session candles", {})

        # ---- indicators on session slice ----
        vwap = _vwap_intraday(day_df)
        ema = _ema(day_df["close"], ema_period)
        atr = _compute_atr(day_df["high"], day_df["low"], day_df["close"], atr_period)

        if pd.isna(atr.iloc[-1]) or atr.iloc[-1] <= 0:
            return Order(symbol, None, None, None, None, "ATR not ready", {})

        atr_now = float(atr.iloc[-1])
        vwap_now = float(vwap.iloc[-1])
        ema_now = float(ema.iloc[-1])
        ema_prev = float(ema.iloc[-2])
        ema_slope_atr = (ema_now - ema_prev) / atr_now

        # regime filter: avoid trends
        if use_regime_filter and abs(ema_slope_atr) > ema_slope_max_atr:
            return Order(symbol, None, None, None, None, "regime: trending (skip MR)", {"ema_slope_atr": ema_slope_atr})

        last = day_df.iloc[-1]
        prev = day_df.iloc[-2]

        last_close = _as_float(last["close"])
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])

        prev_close = _as_float(prev["close"])

        # deviation in ATR units
        dev_down = (vwap_now - last_close) / atr_now
        dev_up = (last_close - vwap_now) / atr_now

        # bands
        band_long = vwap_now - reentry_atr_mult * atr_now
        band_short = vwap_now + reentry_atr_mult * atr_now

        # stretch bands (must have stretched first)
        stretch_long = vwap_now - stretch_atr_mult * atr_now
        stretch_short = vwap_now + stretch_atr_mult * atr_now

        # ---- LONG MR: stretch below, then re-enter band with bullish candle ----
        long_ok = (prev_close <= stretch_long) and (last_close >= band_long) and (last_close < vwap_now)
        long_confirm = (last_close > last_open)  # bullish body
        if long_ok and long_confirm:
            entry = last_close
            stop = min(last_low, entry) - stop_atr_mult * atr_now

            # hard cap stop distance (optional)
            min_stop = entry * (1.0 - max_stop_pct)
            stop = max(stop, min_stop)

            take = vwap_now

            if stop < entry and take > entry:
                st.trades_today += 1
                return Order(
                    symbol=symbol,
                    side="buy",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="VWAP MR long v2 (stretch->reentry, range regime)",
                    meta={
                        "vwap": vwap_now,
                        "atr": atr_now,
                        "ema": ema_now,
                        "ema_slope_atr": ema_slope_atr,
                        "dev_atr": dev_down,
                        "stretch_atr_mult": stretch_atr_mult,
                        "reentry_atr_mult": reentry_atr_mult,
                        "max_stop_pct": max_stop_pct,
                    },
                )

        # ---- SHORT MR: stretch above, then re-enter band with bearish candle ----
        short_ok = (prev_close >= stretch_short) and (last_close <= band_short) and (last_close > vwap_now)
        short_confirm = (last_close < last_open)  # bearish body
        if short_ok and short_confirm:
            entry = last_close
            stop = max(last_high, entry) + stop_atr_mult * atr_now

            # hard cap stop distance (optional)
            max_stop = entry * (1.0 + max_stop_pct)
            stop = min(stop, max_stop)

            take = vwap_now

            if stop > entry and take < entry:
                st.trades_today += 1
                return Order(
                    symbol=symbol,
                    side="sell",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="VWAP MR short v2 (stretch->reentry, range regime)",
                    meta={
                        "vwap": vwap_now,
                        "atr": atr_now,
                        "ema": ema_now,
                        "ema_slope_atr": ema_slope_atr,
                        "dev_atr": dev_up,
                        "stretch_atr_mult": stretch_atr_mult,
                        "reentry_atr_mult": reentry_atr_mult,
                        "max_stop_pct": max_stop_pct,
                    },
                )

        return Order(symbol, None, None, None, None, "no setup", {"dev_down": dev_down, "dev_up": dev_up, "vwap": vwap_now})

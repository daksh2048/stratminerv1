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


def _to_market_tz(ts: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    """
    CRITICAL:
    yfinance intraday often returns tz-naive timestamps that already represent US/Eastern market time.
    If we localize naive -> UTC, we shift the session and kill signals.

    Rule:
      - naive -> localize to market_tz
      - aware -> convert to market_tz
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(market_tz)
    return ts.tz_convert(market_tz)


def _index_to_market_tz(index: pd.DatetimeIndex, market_tz: str) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize(market_tz)
    return idx.tz_convert(market_tz)


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


@dataclass
class RSIState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0
    last_trade_bar_index: Optional[int] = None


class RSIExtremesReversal(Strategy):
    """
    RSI Extremes Reversal (intraday mean reversion)

    Base entry:
      Long:  prev_rsi < oversold and rsi_now >= oversold and last_close > prev_close
      Short: prev_rsi > overbought and rsi_now <= overbought and last_close < prev_close

    Hardening:
      - correct timezone/session slicing
      - min minutes after open
      - cooldown bars
      - optional "extreme" requirement (RSI must pierce deeper than threshold)
      - optional regime filter using EMA slope in ATR units
    """

    def __init__(self, name: str = "rsi_rev", **params) -> None:
        super().__init__(name, **params)
        self._state_by_symbol: dict[str, RSIState] = {}

    def _get_state(self, symbol: str) -> RSIState:
        if symbol not in self._state_by_symbol:
            self._state_by_symbol[symbol] = RSIState()
        return self._state_by_symbol[symbol]

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # ---- params ----
        rsi_period = int(self.params.get("rsi_period", 14))
        oversold = float(self.params.get("oversold", 30.0))
        overbought = float(self.params.get("overbought", 70.0))

        use_extreme_filter = bool(self.params.get("use_extreme_filter", True))
        extreme_buffer = float(self.params.get("extreme_buffer", 3.0))  # need <= (oversold-buffer) or >= (overbought+buffer)

        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_mult", 1.0))
        rr = float(self.params.get("risk_reward", 1.5))

        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_only = bool(self.params.get("rth_only", True))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))

        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 10))
        entry_cutoff_minutes = self.params.get("entry_cutoff_minutes", 300)
        entry_cutoff_minutes = None if entry_cutoff_minutes is None else int(entry_cutoff_minutes)

        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))
        cooldown_bars = int(self.params.get("cooldown_bars", 6))

        use_regime_filter = bool(self.params.get("use_regime_filter", True))
        ema_period = int(self.params.get("ema_period", 50))
        ema_slope_max_atr = float(self.params.get("ema_slope_max_atr", 0.25))

        # broker meta knobs (safe to include; broker ignores if unsupported)
        partial_take_pct = float(self.params.get("partial_take_pct", 0.5))
        runner_take_mode = str(self.params.get("runner_take_mode", "rr")).lower()
        runner_rr = float(self.params.get("runner_rr", 3.0))
        trail_pct = float(self.params.get("trail_pct", 0.003))
        move_stop_to_be = bool(self.params.get("move_stop_to_be", True))

        warmup = max(rsi_period + 5, atr_period + 5, ema_period + 5)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "not enough candles", {})

        # ---- timestamp / day ----
        last_ts = pd.Timestamp(df.index[-1])
        last_ts_m = _to_market_tz(last_ts, market_tz)
        day_m = last_ts_m.normalize()

        # ---- session bounds ----
        open_h, open_m = map(int, rth_open.split(":"))
        close_h, close_m = map(int, rth_close.split(":"))
        session_open = day_m + pd.Timedelta(hours=open_h, minutes=open_m)
        session_close = day_m + pd.Timedelta(hours=close_h, minutes=close_m)

        if rth_only and not (session_open <= last_ts_m <= session_close):
            return Order(symbol, None, None, None, None, "outside RTH", {"ts": str(last_ts_m)})

        if min_minutes_after_open > 0 and last_ts_m < (session_open + pd.Timedelta(minutes=min_minutes_after_open)):
            return Order(symbol, None, None, None, None, "skip: too early after open", {"ts": str(last_ts_m)})

        if entry_cutoff_minutes is not None:
            cutoff_ts = session_open + pd.Timedelta(minutes=entry_cutoff_minutes)
            if last_ts_m > cutoff_ts:
                return Order(symbol, None, None, None, None, "past entry cutoff", {"ts": str(last_ts_m), "cutoff": str(cutoff_ts)})

        # ---- today's RTH slice ----
        idx_m = _index_to_market_tz(pd.DatetimeIndex(df.index), market_tz)
        in_today = (idx_m.normalize() == day_m)
        in_session = (idx_m >= session_open) & (idx_m <= session_close)
        day_rth = df.loc[in_today & in_session].copy()

        if len(day_rth) < warmup:
            return Order(symbol, None, None, None, None, "waiting warmup", {"rth_bars": len(day_rth)})

        # ---- per-day state ----
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
            st.last_trade_bar_index = None

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day reached", {"day": str(day_m)})

        current_bar_index = len(day_rth) - 1
        if st.last_trade_bar_index is not None:
            if (current_bar_index - st.last_trade_bar_index) < cooldown_bars:
                return Order(symbol, None, None, None, None, "cooldown", {"bars_since_trade": current_bar_index - st.last_trade_bar_index})

        close = day_rth["close"].astype(float)
        high = day_rth["high"].astype(float)
        low = day_rth["low"].astype(float)

        rsi = _compute_rsi(close, rsi_period)
        atr = _compute_atr(high, low, close, atr_period)
        ema = _ema(close, ema_period)

        last = day_rth.iloc[-1]
        prev = day_rth.iloc[-2]

        rsi_now = _as_float(rsi.iloc[-1])
        rsi_prev = _as_float(rsi.iloc[-2])

        atr_now = atr.iloc[-1]
        if pd.isna(atr_now) or float(atr_now) <= 0:
            return Order(symbol, None, None, None, None, "ATR not ready", {})
        atr_now = float(atr_now)

        ema_now = float(ema.iloc[-1])
        ema_prev = float(ema.iloc[-2])
        ema_slope_atr = (ema_now - ema_prev) / atr_now

        if use_regime_filter and abs(ema_slope_atr) > ema_slope_max_atr:
            return Order(symbol, None, None, None, None, "regime filter: strong trend", {"ema_slope_atr": ema_slope_atr})

        last_close = _as_float(last["close"])
        prev_close = _as_float(prev["close"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])

        # Extreme requirement
        oversold_extreme = oversold - extreme_buffer
        overbought_extreme = overbought + extreme_buffer

        # ---- LONG ----
        long_reentry = (rsi_prev < oversold) and (rsi_now >= oversold)
        long_confirm = (last_close > prev_close)
        long_extreme_ok = True
        if use_extreme_filter:
            recent_min = float(rsi.iloc[-6:].min())  # ~last 30 mins on 5m
            long_extreme_ok = recent_min <= oversold_extreme

        if long_reentry and long_confirm and long_extreme_ok:
            entry = last_close
            stop = last_low - atr_mult * atr_now
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop", {"rsi": rsi_now, "atr": atr_now})
            take = entry + rr * (entry - stop)

            st.trades_today += 1
            st.last_trade_bar_index = current_bar_index

            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="RSI oversold reversal (re-entry + confirm)",
                meta={
                    "rsi_prev": rsi_prev,
                    "rsi": rsi_now,
                    "oversold": oversold,
                    "atr": atr_now,
                    "ema_slope_atr": ema_slope_atr,
                    "use_extreme_filter": use_extreme_filter,
                    "extreme_buffer": extreme_buffer,
                    "partial_take_pct": partial_take_pct,
                    "runner_take_mode": runner_take_mode,
                    "runner_rr": runner_rr,
                    "trail_pct": trail_pct,
                    "move_stop_to_be": move_stop_to_be,
                    "trail_activate_after_partial": True,
                    "partial_done": False,
                },
            )

        # ---- SHORT ----
        short_reentry = (rsi_prev > overbought) and (rsi_now <= overbought)
        short_confirm = (last_close < prev_close)
        short_extreme_ok = True
        if use_extreme_filter:
            recent_max = float(rsi.iloc[-6:].max())
            short_extreme_ok = recent_max >= overbought_extreme

        if short_reentry and short_confirm and short_extreme_ok:
            entry = last_close
            stop = last_high + atr_mult * atr_now
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop", {"rsi": rsi_now, "atr": atr_now})
            take = entry - rr * (stop - entry)

            st.trades_today += 1
            st.last_trade_bar_index = current_bar_index

            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="RSI overbought reversal (re-entry + confirm)",
                meta={
                    "rsi_prev": rsi_prev,
                    "rsi": rsi_now,
                    "overbought": overbought,
                    "atr": atr_now,
                    "ema_slope_atr": ema_slope_atr,
                    "use_extreme_filter": use_extreme_filter,
                    "extreme_buffer": extreme_buffer,
                    "partial_take_pct": partial_take_pct,
                    "runner_take_mode": runner_take_mode,
                    "runner_rr": runner_rr,
                    "trail_pct": trail_pct,
                    "move_stop_to_be": move_stop_to_be,
                    "trail_activate_after_partial": True,
                    "partial_done": False,
                },
            )

        return Order(symbol, None, None, None, None, "no RSI setup", {"rsi": rsi_now})

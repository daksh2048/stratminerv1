from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _as_float(x) -> float:
    """Convert scalar / 1-elem Series / numpy scalar to float safely."""
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def _col_as_series(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Robustly fetch a column as a Series even if pandas returns a DataFrame
    (e.g., duplicate column names).
    """
    x = df[name]
    if isinstance(x, pd.DataFrame):
        # take first matching column
        x = x.iloc[:, 0]
    return x.astype(float)


def _to_market_tz(ts: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    """
    Treat tz-naive timestamps as already in market time (NOT UTC) to avoid shifting RTH.
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


def _compute_vwap(day_df: pd.DataFrame) -> pd.Series:
    """
    VWAP over the provided slice (expected to be 'today RTH').
    vwap = cum(tp*vol)/cum(vol), tp=(h+l+c)/3
    """
    high = _col_as_series(day_df, "high")
    low = _col_as_series(day_df, "low")
    close = _col_as_series(day_df, "close")

    # volume can be missing or non-numeric in some feeds; handle safely
    vol = day_df["volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    vol = pd.to_numeric(vol, errors="coerce").fillna(0.0)

    tp = (high + low + close) / 3.0

    # avoid division by zero: treat 0 vol as NA so cum vol doesn't stall incorrectly
    vol = vol.replace(0.0, pd.NA)
    cum_pv = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    vwap = cum_pv / cum_v
    return vwap


@dataclass
class RSIState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0
    last_trade_bar_index: Optional[int] = None


class RSIExtremesReversal(Strategy):
    """
    RSI + VWAP deviation mean reversion (intraday)

    Long:
      - RSI <= oversold
      - price <= VWAP - dev_atr_mult*ATR
      - bullish confirmation candle
      - target = VWAP (capped by rr_cap)
    Short: symmetric.
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
        oversold = float(self.params.get("oversold", 35.0))
        overbought = float(self.params.get("overbought", 65.0))

        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_mult", 1.0))

        dev_atr_mult = float(self.params.get("dev_atr_mult", 0.8))
        rr_cap = float(self.params.get("rr_cap", 2.0))

        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_only = bool(self.params.get("rth_only", True))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 10))
        entry_cutoff_minutes = self.params.get("entry_cutoff_minutes", 300)
        entry_cutoff_minutes = None if entry_cutoff_minutes is None else int(entry_cutoff_minutes)

        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))
        cooldown_bars = int(self.params.get("cooldown_bars", 6))

        use_regime_filter = bool(self.params.get("use_regime_filter", False))
        ema_period = int(self.params.get("ema_period", 50))
        ema_slope_max_atr = float(self.params.get("ema_slope_max_atr", 0.6))
        ema_slope_lookback = int(self.params.get("ema_slope_lookback", 5))

        # broker meta (optional)
        partial_take_pct = float(self.params.get("partial_take_pct", 0.5))
        runner_take_mode = str(self.params.get("runner_take_mode", "rr")).lower()
        runner_rr = float(self.params.get("runner_rr", 3.0))
        trail_pct = float(self.params.get("trail_pct", 0.003))
        move_stop_to_be = bool(self.params.get("move_stop_to_be", True))

        warmup = max(rsi_period, atr_period, ema_period) + 10
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "not enough candles", {})

        # ---- last ts in market tz ----
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

        if last_ts_m < session_open + pd.Timedelta(minutes=min_minutes_after_open):
            return Order(symbol, None, None, None, None, "too early after open", {"ts": str(last_ts_m)})

        if entry_cutoff_minutes is not None:
            cutoff_ts = session_open + pd.Timedelta(minutes=entry_cutoff_minutes)
            if last_ts_m > cutoff_ts:
                return Order(symbol, None, None, None, None, "past entry cutoff", {"ts": str(last_ts_m), "cutoff": str(cutoff_ts)})

        # ---- slice today's RTH in market tz ----
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
        if st.last_trade_bar_index is not None and (current_bar_index - st.last_trade_bar_index) < cooldown_bars:
            return Order(symbol, None, None, None, None, "cooldown", {"bars_since_trade": current_bar_index - st.last_trade_bar_index})

        # ---- indicators ----
        close = _col_as_series(day_rth, "close")
        high = _col_as_series(day_rth, "high")
        low = _col_as_series(day_rth, "low")

        rsi = _compute_rsi(close, rsi_period)
        atr = _compute_atr(high, low, close, atr_period)
        ema = _ema(close, ema_period)
        vwap = _compute_vwap(day_rth)

        last = day_rth.iloc[-1]
        prev = day_rth.iloc[-2]

        rsi_now = _as_float(rsi.iloc[-1])
        atr_now = _as_float(atr.iloc[-1])
        vwap_now = _as_float(vwap.iloc[-1])

        if pd.isna(atr_now) or atr_now <= 0 or pd.isna(vwap_now):
            return Order(symbol, None, None, None, None, "indicators not ready", {})

        last_close = _as_float(last["close"])
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        prev_close = _as_float(prev["close"])

        bullish_confirm = (last_close > last_open) and (last_close > prev_close)
        bearish_confirm = (last_close < last_open) and (last_close < prev_close)

        # optional regime filter
        ema_slope_atr = 0.0
        if use_regime_filter:
            lb = min(ema_slope_lookback, len(ema) - 1)
            ema_slope_atr = (_as_float(ema.iloc[-1]) - _as_float(ema.iloc[-1 - lb])) / atr_now
            if abs(ema_slope_atr) > ema_slope_max_atr:
                return Order(symbol, None, None, None, None, "regime filter: strong trend", {"ema_slope_atr": ema_slope_atr})

        dev = dev_atr_mult * atr_now

        # LONG
        if (rsi_now <= oversold) and (last_close <= vwap_now - dev) and bullish_confirm:
            entry = last_close
            stop = last_low - atr_mult * atr_now
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop", {})

            rr_take = entry + rr_cap * (entry - stop)
            take = min(vwap_now, rr_take)
            if take <= entry:
                return Order(symbol, None, None, None, None, "bad long take", {"vwap": vwap_now})

            st.trades_today += 1
            st.last_trade_bar_index = current_bar_index

            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="RSI+VWAP MR long",
                meta={
                    "rsi": rsi_now,
                    "vwap": vwap_now,
                    "atr": atr_now,
                    "dev_atr_mult": dev_atr_mult,
                    "ema_slope_atr": ema_slope_atr,
                    "partial_take_pct": partial_take_pct,
                    "runner_take_mode": runner_take_mode,
                    "runner_rr": runner_rr,
                    "trail_pct": trail_pct,
                    "move_stop_to_be": move_stop_to_be,
                    "trail_activate_after_partial": True,
                    "partial_done": False,
                },
            )

        # SHORT
        if (rsi_now >= overbought) and (last_close >= vwap_now + dev) and bearish_confirm:
            entry = last_close
            stop = last_high + atr_mult * atr_now
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop", {})

            rr_take = entry - rr_cap * (stop - entry)
            take = max(vwap_now, rr_take)
            if take >= entry:
                return Order(symbol, None, None, None, None, "bad short take", {"vwap": vwap_now})

            st.trades_today += 1
            st.last_trade_bar_index = current_bar_index

            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="RSI+VWAP MR short",
                meta={
                    "rsi": rsi_now,
                    "vwap": vwap_now,
                    "atr": atr_now,
                    "dev_atr_mult": dev_atr_mult,
                    "ema_slope_atr": ema_slope_atr,
                    "partial_take_pct": partial_take_pct,
                    "runner_take_mode": runner_take_mode,
                    "runner_rr": runner_rr,
                    "trail_pct": trail_pct,
                    "move_stop_to_be": move_stop_to_be,
                    "trail_activate_after_partial": True,
                    "partial_done": False,
                },
            )

        return Order(symbol, None, None, None, None, "no setup", {"rsi": rsi_now, "vwap": vwap_now})

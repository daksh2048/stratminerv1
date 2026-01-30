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
    yfinance intraday often returns timezone-naive timestamps that already represent
    the market timezone (US/Eastern). If we incorrectly treat them as UTC, the RTH
    filter kills all trades.

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
class VWAPMRState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class VWAPMeanReversion(Strategy):
    """
    VWAP Mean Reversion.

    Default entry logic (TRADES):
      - stretched from VWAP by dev_atr_mult * ATR
      - one-bar reversal confirmation (close vs prev close)
      - regime filter using EMA slope (optional)

    Optional (stricter) entry logic (EDGE upgrade):
      - stretched, then re-enter VWAP±band (reentry_atr_mult * ATR)
      - enable with use_reentry=True
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
        dev_atr_mult = float(self.params.get("dev_atr_mult", 1.2))
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.0))

        # session
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_only = bool(self.params.get("rth_only", True))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 0))  # keep 0 by default

        entry_cutoff_minutes = self.params.get("entry_cutoff_minutes", None)
        entry_cutoff_minutes = None if entry_cutoff_minutes is None else int(entry_cutoff_minutes)

        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))

        # regime filter
        use_regime_filter = bool(self.params.get("use_regime_filter", True))
        ema_period = int(self.params.get("ema_period", 50))
        ema_slope_max_atr = float(self.params.get("ema_slope_max_atr", 0.25))

        # exits / broker meta
        partial_take_pct = float(self.params.get("partial_take_pct", 0.5))
        runner_take_mode = str(self.params.get("runner_take_mode", "rr")).lower()
        runner_rr = float(self.params.get("runner_rr", 3.0))
        trail_pct = float(self.params.get("trail_pct", 0.003))
        move_stop_to_be = bool(self.params.get("move_stop_to_be", True))

        # optional stricter "re-entry" logic
        use_reentry = bool(self.params.get("use_reentry", False))  # OFF by default to ensure trades
        reentry_atr_mult = float(self.params.get("reentry_atr_mult", 0.6))

        warmup = max(atr_period + 20, ema_period + 5)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "not enough candles", {})

        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume for VWAP", {})

        # ---- time + session bounds ----
        last_ts = pd.Timestamp(df.index[-1])
        last_ts_m = _to_market_tz(last_ts, market_tz)
        day_m = last_ts_m.normalize()

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
                return Order(symbol, None, None, None, None, "past entry cutoff", {"ts": str(last_ts_m)})

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

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day reached", {"day": str(day_m)})

        # ---- compute VWAP (session) ----
        vol = day_rth["volume"].astype(float)
        if float(vol.sum()) <= 0:
            return Order(symbol, None, None, None, None, "zero volume", {})

        typical = (day_rth["high"].astype(float) + day_rth["low"].astype(float) + day_rth["close"].astype(float)) / 3.0
        vwap = (typical * vol).cumsum() / vol.cumsum()

        # ---- ATR / EMA ----
        high = day_rth["high"].astype(float)
        low = day_rth["low"].astype(float)
        close = day_rth["close"].astype(float)

        atr = _compute_atr(high, low, close, atr_period)
        ema = _ema(close, ema_period)

        if len(day_rth) < 3:
            return Order(symbol, None, None, None, None, "not enough RTH bars", {})

        last = day_rth.iloc[-1]
        prev = day_rth.iloc[-2]

        vwap_now = _as_float(vwap.iloc[-1])
        atr_now = atr.iloc[-1]
        if pd.isna(atr_now) or float(atr_now) <= 0:
            return Order(symbol, None, None, None, None, "ATR not ready", {})

        atr_now = float(atr_now)
        last_close = _as_float(last["close"])
        prev_close = _as_float(prev["close"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])

        ema_now = float(ema.iloc[-1])
        ema_prev = float(ema.iloc[-2])
        ema_slope_atr = (ema_now - ema_prev) / atr_now

        # regime filter
        if use_regime_filter and abs(ema_slope_atr) > ema_slope_max_atr:
            return Order(symbol, None, None, None, None, "regime filter: strong trend", {"ema_slope_atr": ema_slope_atr})

        # deviation in ATR units
        dev_down = (vwap_now - last_close) / atr_now  # + means below VWAP
        dev_up = (last_close - vwap_now) / atr_now    # + means above VWAP

        # optional re-entry bands
        band_long = vwap_now - reentry_atr_mult * atr_now
        band_short = vwap_now + reentry_atr_mult * atr_now

        # -------------------------
        # LONG MR
        # -------------------------
        long_ok = dev_down >= dev_atr_mult
        if long_ok:
            if use_reentry:
                # stretched + re-enter above band
                long_trigger = (prev_close < band_long) and (last_close >= band_long)
            else:
                # stretched + reversal candle
                long_trigger = (last_close > prev_close)

            if long_trigger and last_close < vwap_now:
                entry = last_close
                stop = last_low - stop_atr_mult * atr_now
                take = vwap_now

                if stop < entry and take > entry:
                    st.trades_today += 1
                    return Order(
                        symbol=symbol,
                        side="buy",
                        entry=entry,
                        stop=stop,
                        take=take,
                        reason="VWAP MR long",
                        meta={
                            "vwap": vwap_now,
                            "atr": atr_now,
                            "ema": ema_now,
                            "ema_slope_atr": ema_slope_atr,
                            "dev_atr": dev_down,
                            "dev_atr_mult": dev_atr_mult,
                            "use_reentry": use_reentry,
                            "band_long": band_long,
                            "partial_take_pct": partial_take_pct,
                            "runner_take_mode": runner_take_mode,
                            "runner_rr": runner_rr,
                            "trail_pct": trail_pct,
                            "move_stop_to_be": move_stop_to_be,
                            "trail_activate_after_partial": True,
                            "partial_done": False,
                        },
                    )

        # -------------------------
        # SHORT MR
        # -------------------------
        short_ok = dev_up >= dev_atr_mult
        if short_ok:
            if use_reentry:
                short_trigger = (prev_close > band_short) and (last_close <= band_short)
            else:
                short_trigger = (last_close < prev_close)

            if short_trigger and last_close > vwap_now:
                entry = last_close
                stop = last_high + stop_atr_mult * atr_now
                take = vwap_now

                if stop > entry and take < entry:
                    st.trades_today += 1
                    return Order(
                        symbol=symbol,
                        side="sell",
                        entry=entry,
                        stop=stop,
                        take=take,
                        reason="VWAP MR short",
                        meta={
                            "vwap": vwap_now,
                            "atr": atr_now,
                            "ema": ema_now,
                            "ema_slope_atr": ema_slope_atr,
                            "dev_atr": dev_up,
                            "dev_atr_mult": dev_atr_mult,
                            "use_reentry": use_reentry,
                            "band_short": band_short,
                            "partial_take_pct": partial_take_pct,
                            "runner_take_mode": runner_take_mode,
                            "runner_rr": runner_rr,
                            "trail_pct": trail_pct,
                            "move_stop_to_be": move_stop_to_be,
                            "trail_activate_after_partial": True,
                            "partial_done": False,
                        },
                    )

        return Order(
            symbol, None, None, None, None,
            "no setup",
            {"dev_down": dev_down, "dev_up": dev_up, "vwap": vwap_now},
        )

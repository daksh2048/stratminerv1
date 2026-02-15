from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd

from src.core.types import Order
from .base import Strategy


def _as_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def _to_market_tz(idx: pd.DatetimeIndex, market_tz: str) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(market_tz)


def _session_bounds(day_m: pd.Timestamp, rth_open: str, rth_close: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    oh, om = map(int, rth_open.split(":"))
    ch, cm = map(int, rth_close.split(":"))
    session_open = day_m + pd.Timedelta(hours=oh, minutes=om)
    session_close = day_m + pd.Timedelta(hours=ch, minutes=cm)
    return session_open, session_close


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _vwap_intraday(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].astype(float)
    v = df["volume"].astype(float).replace(0, pd.NA)
    vwap = pv.cumsum() / v.cumsum()
    return vwap.ffill().infer_objects(copy=False)


@dataclass
class MomoState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class VWAPMomentum(Strategy):
    """
    VWAP Momentum Continuation - Institutional Quality
    
    Logic:
      1. Strong trend: Price > VWAP > EMA (or inverted for short)
      2. EMA slope strong (>0.15 ATR/bar for long, <-0.15 for short)
      3. Pullback: Price touches VWAP WITHOUT crossing it
      4. Continuation bar: Strong directional bar with volume
      5. Enter on continuation close, stop below VWAP
    
    Key Improvements:
      - MUCH tighter trend definition (slope requirement)
      - Pullback must TOUCH vwap, not cross (stays in trend)
      - Requires VOLUME on continuation bar
      - Requires STRONG continuation bar (>66% of ATR)
      - Clean stop below VWAP (not arbitrary)
    """

    def __init__(self, name: str = "vwap_momentum", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, MomoState] = {}

    def _get_state(self, symbol: str) -> MomoState:
        if symbol not in self._state:
            self._state[symbol] = MomoState()
        return self._state[symbol]

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 30))
        entry_cutoff_minutes = int(self.params.get("entry_cutoff_minutes", 330))
        
        ema_period = int(self.params.get("ema_period", 20))
        atr_period = int(self.params.get("atr_period", 14))
        
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.0))
        risk_reward = float(self.params.get("risk_reward", 2.5))
        
        # Trend strength: EMA slope in ATR units
        min_ema_slope_atr = float(self.params.get("min_ema_slope_atr", 0.15))
        
        # Pullback tolerance: how close to VWAP (in ATR)
        pullback_tolerance_atr = float(self.params.get("pullback_tolerance_atr", 0.3))
        
        # Continuation bar: minimum size (in ATR)
        min_continuation_atr = float(self.params.get("min_continuation_atr", 0.66))
        
        # Volume
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.5))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 3))

        warmup = max(ema_period, atr_period, volume_period) + 5
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume", {})

        # --- timezone ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        session_open, session_close = _session_bounds(day_m, rth_open, rth_close)

        in_today = idx_m.normalize() == day_m
        day_df = df.loc[in_today].copy()
        
        if day_df.empty or len(day_df) < warmup:
            return Order(symbol, None, None, None, None, "not enough bars today", {})

        # --- time gates ---
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        if last_ts_m < (session_open + pd.Timedelta(minutes=min_minutes_after_open)):
            return Order(symbol, None, None, None, None, "too early", {})

        if last_ts_m > (session_open + pd.Timedelta(minutes=entry_cutoff_minutes)):
            return Order(symbol, None, None, None, None, "past cutoff", {})

        # --- state ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})

        # --- indicators ---
        vwap = _vwap_intraday(day_df)
        ema = _ema(day_df["close"].astype(float), ema_period)
        atr = _atr(
            day_df["high"].astype(float),
            day_df["low"].astype(float),
            day_df["close"].astype(float),
            atr_period
        )
        
        vol_avg = day_df["volume"].astype(float).rolling(volume_period).mean()

        if pd.isna(vwap.iloc[-1]) or pd.isna(ema.iloc[-1]) or pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})

        vwap_now = _as_float(vwap.iloc[-1])
        ema_now = _as_float(ema.iloc[-1])
        ema_prev = _as_float(ema.iloc[-2])
        atr_now = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])

        if atr_now <= 0 or vol_avg_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})

        # Current bar
        last = day_df.iloc[-1]
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        last_close = _as_float(last["close"])
        last_volume = _as_float(last["volume"])

        # Previous bar
        prev = day_df.iloc[-2]
        prev_low = _as_float(prev["low"])
        prev_high = _as_float(prev["high"])

        # --- TREND STRENGTH CHECK ---
        ema_slope_atr = (ema_now - ema_prev) / atr_now
        
        strong_uptrend = ema_slope_atr >= min_ema_slope_atr
        strong_downtrend = ema_slope_atr <= -min_ema_slope_atr

        # --- TREND ALIGNMENT ---
        # For uptrend: close > VWAP > EMA
        uptrend_aligned = (last_close > vwap_now) and (vwap_now > ema_now)
        
        # For downtrend: close < VWAP < EMA
        downtrend_aligned = (last_close < vwap_now) and (vwap_now < ema_now)

        # --- PULLBACK: Touched VWAP without crossing ---
        tolerance = pullback_tolerance_atr * atr_now
        
        # Long pullback: previous bar touched near VWAP from above, stayed above
        long_pullback = (prev_low <= vwap_now + tolerance) and (prev_low > vwap_now - tolerance)
        
        # Short pullback: previous bar touched near VWAP from below, stayed below
        short_pullback = (prev_high >= vwap_now - tolerance) and (prev_high < vwap_now + tolerance)

        # --- CONTINUATION BAR: Strong directional move with volume ---
        bar_size = abs(last_close - last_open)
        strong_bar = bar_size >= (min_continuation_atr * atr_now)
        
        bullish_bar = last_close > last_open
        bearish_bar = last_close < last_open
        
        volume_confirmation = last_volume >= (volume_mult * vol_avg_now)

        # --- LONG SETUP ---
        if strong_uptrend and uptrend_aligned and long_pullback and bullish_bar and strong_bar and volume_confirmation:
            entry = last_close
            stop = vwap_now - (stop_atr_mult * atr_now)
            
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop", {"stop": stop, "entry": entry})
            
            take = entry + risk_reward * (entry - stop)
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="VWAP momentum long",
                meta={
                    "vwap": vwap_now,
                    "ema": ema_now,
                    "atr": atr_now,
                    "ema_slope_atr": ema_slope_atr,
                    "bar_size_atr": bar_size / atr_now,
                    "volume_ratio": last_volume / vol_avg_now,
                    "trail_mode": self.params.get("trail_mode", "tiered"),
                    "trail_activate_after_partial": True,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 1.5,
                    "move_stop_to_be": True,
                    "runner_take_mode": "rr",
                    "runner_rr": risk_reward,
                    "initial_stop": stop,
                },
            )

        # --- SHORT SETUP ---
        if strong_downtrend and downtrend_aligned and short_pullback and bearish_bar and strong_bar and volume_confirmation:
            entry = last_close
            stop = vwap_now + (stop_atr_mult * atr_now)
            
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop", {"stop": stop, "entry": entry})
            
            take = entry - risk_reward * (stop - entry)
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="VWAP momentum short",
                meta={
                    "vwap": vwap_now,
                    "ema": ema_now,
                    "atr": atr_now,
                    "ema_slope_atr": ema_slope_atr,
                    "bar_size_atr": bar_size / atr_now,
                    "volume_ratio": last_volume / vol_avg_now,
                    "trail_mode": self.params.get("trail_mode", "tiered"),
                    "trail_activate_after_partial": True,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 1.5,
                    "move_stop_to_be": True,
                    "runner_take_mode": "rr",
                    "runner_rr": risk_reward,
                    "initial_stop": stop,
                },
            )

        return Order(symbol, None, None, None, None, "no momentum setup", {})
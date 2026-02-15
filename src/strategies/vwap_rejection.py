from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def _session_bounds(day_m: pd.Timestamp, rth_open: str, rth_close: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    oh, om = map(int, rth_open.split(":"))
    ch, cm = map(int, rth_close.split(":"))
    session_open = day_m + pd.Timedelta(hours=oh, minutes=om)
    session_close = day_m + pd.Timedelta(hours=ch, minutes=cm)
    return session_open, session_close


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _vwap_session(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].astype(float)
    v = df["volume"].astype(float).replace(0.0, pd.NA)
    cum_pv = pv.cumsum()
    cum_v = v.cumsum()
    vwap = (cum_pv / cum_v)
    return vwap.ffill().infer_objects(copy=False)


@dataclass
class _DayState:
    last_day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class VWAPRejection(Strategy):
    """
    VWAP Rejection (Mean Reversion) - Institutional Quality
    
    Logic:
      1. Price extends beyond outer band (VWAP ± band_mult * ATR)
      2. Rejection bar forms: reversal candle with volume spike
      3. Enter on rejection close, stop beyond wick
      4. Target: VWAP (mean reversion)
    
    Key Improvements vs old version:
      - Requires VOLUME SPIKE on rejection bar (2x avg volume)
      - Requires REVERSAL CANDLE (long wick in direction of stretch)
      - Single-bar entry (no state machine leak)
      - Tighter stop (beyond rejection wick, not just band)
      - Clear target (VWAP only, no RR fallback)
    """

    def __init__(self, name: str = "vwap_rejection", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, _DayState] = {}

    def _get_state(self, symbol: str) -> _DayState:
        if symbol not in self._state:
            self._state[symbol] = _DayState()
        return self._state[symbol]

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        atr_period = int(self.params.get("atr_period", 14))
        band_atr_mult = float(self.params.get("band_atr_mult", 2.0))  # Wider bands = fewer setups
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 30))
        max_minutes_after_open = int(self.params.get("max_minutes_after_open", 360))
        
        stop_buffer_atr = float(self.params.get("stop_buffer_atr", 0.3))
        
        # Volume spike threshold
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 2.0))
        
        # Reversal candle: wick must be X% of total range
        min_wick_pct = float(self.params.get("min_wick_pct", 0.4))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 3))

        warmup = max(atr_period, volume_period) + 5
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume", {})

        # --- timezone conversion ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        session_open, session_close = _session_bounds(day_m, rth_open, rth_close)

        # Calculate on today's data
        in_today = idx_m.normalize() == day_m
        day_df = df.loc[in_today].copy()
        
        if day_df.empty or len(day_df) < warmup:
            return Order(symbol, None, None, None, None, "not enough bars today", {})

        # --- time window ---
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        mins_from_open = (last_ts_m - session_open).total_seconds() / 60.0
        if mins_from_open < min_minutes_after_open:
            return Order(symbol, None, None, None, None, "too early", {})
        if mins_from_open > max_minutes_after_open:
            return Order(symbol, None, None, None, None, "past window", {})

        # --- state ---
        st = self._get_state(symbol)
        if st.last_day is None or st.last_day != day_m:
            st.last_day = day_m
            st.trades_today = 0

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})

        # --- indicators ---
        vwap = _vwap_session(day_df)
        atr = _atr(day_df, atr_period)
        
        # Volume average
        vol_avg = day_df["volume"].astype(float).rolling(volume_period).mean()

        if pd.isna(vwap.iloc[-1]) or pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})

        v = _as_float(vwap.iloc[-1])
        a = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])

        if a <= 0 or vol_avg_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})

        upper_band = v + band_atr_mult * a
        lower_band = v - band_atr_mult * a

        # Current bar
        last = day_df.iloc[-1]
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        last_close = _as_float(last["close"])
        last_volume = _as_float(last["volume"])

        bar_range = last_high - last_low
        if bar_range <= 0:
            return Order(symbol, None, None, None, None, "zero range bar", {})

        # --- LONG REJECTION: Price stretched below, rejection bar forms ---
        if last_low < lower_band:  # Touched below band
            
            # Check: Rejection candle (bullish close with lower wick)
            lower_wick = last_close - last_low  # How much wick below close
            wick_pct = lower_wick / bar_range
            
            bullish_close = last_close > last_open
            has_wick = wick_pct >= min_wick_pct
            
            # Check: Volume spike
            volume_spike = last_volume >= (volume_mult * vol_avg_now)
            
            if bullish_close and has_wick and volume_spike:
                entry = last_close
                stop = last_low - (stop_buffer_atr * a)  # Below rejection wick
                take = v  # Target VWAP
                
                if stop >= entry or take <= entry:
                    return Order(symbol, None, None, None, None, "invalid long levels", 
                               {"stop": stop, "entry": entry, "take": take})
                
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="buy",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="VWAP rejection long",
                    meta={
                        "vwap": v,
                        "atr": a,
                        "lower_band": lower_band,
                        "wick_pct": wick_pct,
                        "volume_ratio": last_volume / vol_avg_now,
                        "distance_to_vwap": (v - entry) / a,
                        "trail_mode": self.params.get("trail_mode", "tiered"),
                        "trail_activate_after_partial": False,  # No partial TP for MR
                        "partial_take_pct": 0.0,  # All or nothing
                    },
                )

        # --- SHORT REJECTION: Price stretched above, rejection bar forms ---
        if last_high > upper_band:  # Touched above band
            
            # Check: Rejection candle (bearish close with upper wick)
            upper_wick = last_high - last_close
            wick_pct = upper_wick / bar_range
            
            bearish_close = last_close < last_open
            has_wick = wick_pct >= min_wick_pct
            
            # Check: Volume spike
            volume_spike = last_volume >= (volume_mult * vol_avg_now)
            
            if bearish_close and has_wick and volume_spike:
                entry = last_close
                stop = last_high + (stop_buffer_atr * a)  # Above rejection wick
                take = v  # Target VWAP
                
                if stop <= entry or take >= entry:
                    return Order(symbol, None, None, None, None, "invalid short levels",
                               {"stop": stop, "entry": entry, "take": take})
                
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="sell",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="VWAP rejection short",
                    meta={
                        "vwap": v,
                        "atr": a,
                        "upper_band": upper_band,
                        "wick_pct": wick_pct,
                        "volume_ratio": last_volume / vol_avg_now,
                        "distance_to_vwap": (entry - v) / a,
                        "trail_mode": self.params.get("trail_mode", "tiered"),
                        "trail_activate_after_partial": False,
                        "partial_take_pct": 0.0,
                    },
                )

        return Order(symbol, None, None, None, None, "no rejection setup", {})
"""
RSI Mean Reversion Strategy
Best for: Oscillating instruments (TZA, TNA, SOXL during choppy markets)

Logic:
1. RSI extreme (< 30 or > 70)
2. Price below/above VWAP (confirmation)
3. Reversal bar with long wick
4. Volume spike
5. Enter on bounce

Why it works:
- RSI extremes mark exhaustion
- VWAP acts as mean
- Wick shows rejection
- Volume confirms institutional reversal
"""
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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _vwap_session(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].astype(float)
    v = df["volume"].astype(float).replace(0.0, pd.NA)
    cum_pv = pv.cumsum()
    cum_v = v.cumsum()
    vwap = (cum_pv / cum_v)
    return vwap.ffill().infer_objects(copy=False)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


@dataclass
class RSIState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class RSIMeanReversion(Strategy):
    """
    RSI Mean Reversion with VWAP Confirmation
    
    Entry Requirements:
    1. RSI extreme (<30 for longs, >70 for shorts)
    2. Price below VWAP (long) or above VWAP (short)
    3. Reversal bar (long wick showing rejection)
    4. Volume >1.5x average
    5. Confirmation bar (close towards mean)
    
    Stop: Beyond wick + buffer
    Target: VWAP or RSI 50
    """
    
    def __init__(self, name: str = "rsi_mean_reversion", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, RSIState] = {}
    
    def _get_state(self, symbol: str) -> RSIState:
        if symbol not in self._state:
            self._state[symbol] = RSIState()
        return self._state[symbol]
    
    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 30))
        max_minutes_after_open = int(self.params.get("max_minutes_after_open", 360))
        
        rsi_period = int(self.params.get("rsi_period", 14))
        rsi_oversold = float(self.params.get("rsi_oversold", 30))
        rsi_overbought = float(self.params.get("rsi_overbought", 70))
        
        atr_period = int(self.params.get("atr_period", 14))
        stop_buffer_atr = float(self.params.get("stop_buffer_atr", 0.5))
        
        min_wick_pct = float(self.params.get("min_wick_pct", 0.3))  # Wick must be 30% of range
        
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.5))
        
        require_vwap_confirmation = bool(self.params.get("require_vwap_confirmation", True))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 3))
        
        warmup = max(rsi_period, atr_period, volume_period) + 5
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})
        
        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume", {})
        
        # --- timezone ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        session_open, session_close = _session_bounds(day_m, rth_open, rth_close)
        
        # Get today's data
        in_today = idx_m.normalize() == day_m
        day_df = df.loc[in_today].copy()
        
        if day_df.empty or len(day_df) < warmup:
            return Order(symbol, None, None, None, None, "not enough bars today", {})
        
        # --- time gates ---
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        mins_from_open = (last_ts_m - session_open).total_seconds() / 60.0
        if mins_from_open < min_minutes_after_open:
            return Order(symbol, None, None, None, None, "too early", {})
        if mins_from_open > max_minutes_after_open:
            return Order(symbol, None, None, None, None, "past window", {})
        
        # --- state ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
        
        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})
        
        # --- indicators ---
        close = df["close"].astype(float)
        rsi = _rsi(close, rsi_period)
        
        vwap = _vwap_session(day_df)
        atr = _atr(df, atr_period)
        vol_avg = df["volume"].astype(float).rolling(volume_period).mean()
        
        if pd.isna(rsi.iloc[-1]) or pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})
        
        rsi_now = _as_float(rsi.iloc[-1])
        atr_now = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])
        
        if atr_now <= 0 or vol_avg_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})
        
        # VWAP (if available for today)
        vwap_now = None
        if not pd.isna(vwap.iloc[-1]):
            vwap_now = _as_float(vwap.iloc[-1])
        
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
        
        # --- LONG SETUP (RSI Oversold) ---
        if rsi_now < rsi_oversold:
            # Check VWAP confirmation (price below VWAP)
            if require_vwap_confirmation and vwap_now is not None:
                if last_close > vwap_now:
                    return Order(symbol, None, None, None, None, "price above VWAP (need below for long)", {})
            
            # Check reversal bar (lower wick >30% of range)
            lower_wick = last_close - last_low if last_close > last_open else last_open - last_low
            wick_pct = lower_wick / bar_range
            
            if wick_pct < min_wick_pct:
                return Order(symbol, None, None, None, None, "insufficient lower wick", {})
            
            # Check bullish close (towards mean)
            if last_close <= last_open:
                return Order(symbol, None, None, None, None, "bearish candle, need bullish", {})
            
            # Check volume
            if last_volume < (volume_mult * vol_avg_now):
                return Order(symbol, None, None, None, None, "insufficient volume", {})
            
            # ENTRY
            entry = last_close
            stop = last_low - (stop_buffer_atr * atr_now)
            
            # Target: VWAP if available, else RSI 50 level (estimate)
            if vwap_now is not None:
                take = vwap_now
            else:
                # Estimate distance to RSI 50 (rough approximation)
                take = entry + (2.0 * atr_now)
            
            if stop >= entry or take <= entry:
                return Order(symbol, None, None, None, None, "invalid long levels", {})
            
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"RSI oversold bounce (RSI {rsi_now:.1f})",
                meta={
                    "rsi": rsi_now,
                    "vwap": vwap_now,
                    "wick_pct": wick_pct,
                    "volume_ratio": last_volume / vol_avg_now,
                    "atr": atr_now,
                    "trail_mode": self.params.get("trail_mode", "percent"),
                    "trail_pct": 0.015,  # Tight trail for MR
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 0.8,
                    "initial_stop": stop,
                }
            )
        
        # --- SHORT SETUP (RSI Overbought) ---
        if rsi_now > rsi_overbought:
            # Check VWAP confirmation (price above VWAP)
            if require_vwap_confirmation and vwap_now is not None:
                if last_close < vwap_now:
                    return Order(symbol, None, None, None, None, "price below VWAP (need above for short)", {})
            
            # Check reversal bar (upper wick >30% of range)
            upper_wick = last_high - last_close if last_close < last_open else last_high - last_open
            wick_pct = upper_wick / bar_range
            
            if wick_pct < min_wick_pct:
                return Order(symbol, None, None, None, None, "insufficient upper wick", {})
            
            # Check bearish close
            if last_close >= last_open:
                return Order(symbol, None, None, None, None, "bullish candle, need bearish", {})
            
            # Check volume
            if last_volume < (volume_mult * vol_avg_now):
                return Order(symbol, None, None, None, None, "insufficient volume", {})
            
            # ENTRY
            entry = last_close
            stop = last_high + (stop_buffer_atr * atr_now)
            
            # Target: VWAP if available
            if vwap_now is not None:
                take = vwap_now
            else:
                take = entry - (2.0 * atr_now)
            
            if stop <= entry or take >= entry:
                return Order(symbol, None, None, None, None, "invalid short levels", {})
            
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"RSI overbought rejection (RSI {rsi_now:.1f})",
                meta={
                    "rsi": rsi_now,
                    "vwap": vwap_now,
                    "wick_pct": wick_pct,
                    "volume_ratio": last_volume / vol_avg_now,
                    "atr": atr_now,
                    "trail_mode": self.params.get("trail_mode", "percent"),
                    "trail_pct": 0.015,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 0.8,
                    "initial_stop": stop,
                }
            )
        
        return Order(symbol, None, None, None, None, "RSI not extreme", {})
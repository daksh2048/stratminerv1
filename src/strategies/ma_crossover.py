"""
Moving Average Crossover Strategy - Trend Following
Best for: Trending instruments (TECL, SOXL, TNA during bull markets)

Logic:
1. Fast MA (20) crosses above/below Slow MA (50)
2. HTF trend confirmation (price > HTF EMA)
3. Pullback to fast MA after cross
4. Volume confirmation
5. Enter on continuation

Why it works:
- MA crossovers identify trend changes
- Pullback reduces entry risk
- HTF filter prevents choppy whipsaws
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


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


@dataclass
class MACrossState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0
    last_cross: Optional[str] = None  # "bullish" or "bearish"
    cross_bar_idx: Optional[int] = None


class MovingAverageCrossover(Strategy):
    """
    MA Crossover with Pullback Entry
    
    Entry Requirements:
    1. Fast MA crosses Slow MA (trend change)
    2. HTF trend confirmation
    3. Price pulls back to Fast MA
    4. Volume > average
    5. Continuation bar in trend direction
    
    Stop: Below/above Slow MA
    Target: 3R
    """
    
    def __init__(self, name: str = "ma_crossover", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, MACrossState] = {}
        self._htf_data: Optional[pd.DataFrame] = None
    
    def set_htf_data(self, htf_df: pd.DataFrame):
        """Set higher timeframe data for trend filter"""
        self._htf_data = htf_df
    
    def _get_state(self, symbol: str) -> MACrossState:
        if symbol not in self._state:
            self._state[symbol] = MACrossState()
        return self._state[symbol]
    
    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 60))
        entry_cutoff_minutes = int(self.params.get("entry_cutoff_minutes", 330))
        
        fast_period = int(self.params.get("fast_period", 20))
        slow_period = int(self.params.get("slow_period", 50))
        
        htf_ema_period = int(self.params.get("htf_ema_period", 20))
        
        atr_period = int(self.params.get("atr_period", 14))
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.5))
        risk_reward = float(self.params.get("risk_reward", 3.0))
        
        pullback_tolerance_atr = float(self.params.get("pullback_tolerance_atr", 0.5))
        min_continuation_pct = float(self.params.get("min_continuation_pct", 0.5))  # Close in top/bottom 50% of bar
        
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.2))
        
        max_bars_since_cross = int(self.params.get("max_bars_since_cross", 20))
        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))
        
        warmup = max(fast_period, slow_period, atr_period, volume_period) + 5
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
        if mins_from_open > entry_cutoff_minutes:
            return Order(symbol, None, None, None, None, "past cutoff", {})
        
        # --- state ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
            st.last_cross = None
            st.cross_bar_idx = None
        
        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})
        
        # --- HTF trend filter ---
        htf_bias = None
        if self._htf_data is not None and not self._htf_data.empty:
            htf_ema = _ema(self._htf_data["close"].astype(float), htf_ema_period)
            if not pd.isna(htf_ema.iloc[-1]):
                htf_close = _as_float(self._htf_data["close"].iloc[-1])
                htf_ema_val = _as_float(htf_ema.iloc[-1])
                
                if htf_close > htf_ema_val:
                    htf_bias = "long"
                elif htf_close < htf_ema_val:
                    htf_bias = "short"
        
        # --- indicators ---
        close = df["close"].astype(float)
        fast_ma = _ema(close, fast_period)
        slow_ma = _ema(close, slow_period)
        
        atr = _atr(df, atr_period)
        vol_avg = df["volume"].astype(float).rolling(volume_period).mean()
        
        if pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]) or pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})
        
        fast_now = _as_float(fast_ma.iloc[-1])
        slow_now = _as_float(slow_ma.iloc[-1])
        atr_now = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])
        
        if atr_now <= 0 or vol_avg_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})
        
        # Current bar
        last = df.iloc[-1]
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        last_close = _as_float(last["close"])
        last_volume = _as_float(last["volume"])
        
        # Previous MA values
        fast_prev = _as_float(fast_ma.iloc[-2])
        slow_prev = _as_float(slow_ma.iloc[-2])
        
        # --- DETECT CROSSOVER ---
        bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)
        
        if bullish_cross and st.last_cross != "bullish":
            st.last_cross = "bullish"
            st.cross_bar_idx = len(df) - 1
            return Order(symbol, None, None, None, None, "bullish cross detected, waiting for pullback", {})
        
        if bearish_cross and st.last_cross != "bearish":
            st.last_cross = "bearish"
            st.cross_bar_idx = len(df) - 1
            return Order(symbol, None, None, None, None, "bearish cross detected, waiting for pullback", {})
        
        # --- CHECK IF CROSS IS TOO OLD ---
        if st.last_cross is not None and st.cross_bar_idx is not None:
            bars_since_cross = len(df) - st.cross_bar_idx
            if bars_since_cross > max_bars_since_cross:
                st.last_cross = None
                st.cross_bar_idx = None
                return Order(symbol, None, None, None, None, "cross expired", {})
        
        # --- WAIT FOR PULLBACK + CONTINUATION ---
        if st.last_cross == "bullish":
            # Check HTF bias
            if htf_bias == "short":
                return Order(symbol, None, None, None, None, "HTF bearish, skip long", {})
            
            # Check pullback: price should have touched near fast MA
            pullback_target = fast_now
            pullback_tolerance = pullback_tolerance_atr * atr_now
            
            # Check if price pulled back (low touched near fast MA)
            if last_low > (pullback_target + pullback_tolerance):
                return Order(symbol, None, None, None, None, "waiting for pullback to fast MA", {})
            
            # Check continuation: bullish bar with close in top half
            bar_range = last_high - last_low
            if bar_range <= 0:
                return Order(symbol, None, None, None, None, "zero range bar", {})
            
            close_position = (last_close - last_low) / bar_range
            
            if last_close <= last_open or close_position < min_continuation_pct:
                return Order(symbol, None, None, None, None, "weak continuation bar", {})
            
            # Check volume
            if last_volume < (volume_mult * vol_avg_now):
                return Order(symbol, None, None, None, None, "insufficient volume", {})
            
            # ENTRY
            entry = last_close
            stop = slow_now - (stop_atr_mult * atr_now)
            
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop", {})
            
            take = entry + risk_reward * (entry - stop)
            st.trades_today += 1
            st.last_cross = None  # Reset to prevent re-entry
            
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"MA crossover long (pullback entry)",
                meta={
                    "fast_ma": fast_now,
                    "slow_ma": slow_now,
                    "htf_bias": htf_bias,
                    "bars_since_cross": len(df) - st.cross_bar_idx,
                    "volume_ratio": last_volume / vol_avg_now,
                    "atr": atr_now,
                    "trail_mode": self.params.get("trail_mode", "tiered"),
                    "trail_activate_after_partial": True,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 1.5,
                    "initial_stop": stop,
                }
            )
        
        elif st.last_cross == "bearish":
            # Check HTF bias
            if htf_bias == "long":
                return Order(symbol, None, None, None, None, "HTF bullish, skip short", {})
            
            # Check pullback
            pullback_target = fast_now
            pullback_tolerance = pullback_tolerance_atr * atr_now
            
            if last_high < (pullback_target - pullback_tolerance):
                return Order(symbol, None, None, None, None, "waiting for pullback to fast MA", {})
            
            # Check continuation: bearish bar with close in bottom half
            bar_range = last_high - last_low
            if bar_range <= 0:
                return Order(symbol, None, None, None, None, "zero range bar", {})
            
            close_position = (last_close - last_low) / bar_range
            
            if last_close >= last_open or close_position > (1.0 - min_continuation_pct):
                return Order(symbol, None, None, None, None, "weak continuation bar", {})
            
            # Check volume
            if last_volume < (volume_mult * vol_avg_now):
                return Order(symbol, None, None, None, None, "insufficient volume", {})
            
            # ENTRY
            entry = last_close
            stop = slow_now + (stop_atr_mult * atr_now)
            
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop", {})
            
            take = entry - risk_reward * (stop - entry)
            st.trades_today += 1
            st.last_cross = None
            
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"MA crossover short (pullback entry)",
                meta={
                    "fast_ma": fast_now,
                    "slow_ma": slow_now,
                    "htf_bias": htf_bias,
                    "bars_since_cross": len(df) - st.cross_bar_idx,
                    "volume_ratio": last_volume / vol_avg_now,
                    "atr": atr_now,
                    "trail_mode": self.params.get("trail_mode", "tiered"),
                    "trail_activate_after_partial": True,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 1.5,
                    "initial_stop": stop,
                }
            )
        
        return Order(symbol, None, None, None, None, "no MA crossover setup", {})
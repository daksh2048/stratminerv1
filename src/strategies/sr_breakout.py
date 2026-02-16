"""
Support/Resistance Breakout Strategy
Best for: All leveraged ETFs with clear levels (BOIL, UCO, TECL, SOXL, TZA)

Logic:
1. Identify key S/R levels (pivot points, previous highs/lows)
2. Price tests level multiple times
3. Breakout with volume
4. Retest (optional)
5. Continuation

Why it works:
- S/R levels are self-fulfilling (everyone watches them)
- Multiple tests show significance
- Volume confirms institutional breakout
- Retest provides low-risk entry
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

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


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _find_pivot_levels(df: pd.DataFrame, lookback: int, tolerance_pct: float) -> List[float]:
    """
    Find pivot highs and lows (S/R levels)
    
    A pivot high is a high that is higher than X bars on each side
    A pivot low is a low that is lower than X bars on each side
    """
    levels = []
    window = 5  # Bars on each side
    
    highs = df["high"].values
    lows = df["low"].values
    
    # Find pivot highs
    for i in range(window, len(highs) - window):
        left_highs = highs[i-window:i]
        right_highs = highs[i+1:i+window+1]
        
        if highs[i] > max(left_highs.max(), right_highs.max()):
            levels.append(float(highs[i]))
    
    # Find pivot lows
    for i in range(window, len(lows) - window):
        left_lows = lows[i-window:i]
        right_lows = lows[i+1:i+window+1]
        
        if lows[i] < min(left_lows.min(), right_lows.min()):
            levels.append(float(lows[i]))
    
    # Cluster nearby levels (within tolerance)
    if not levels:
        return []
    
    levels.sort()
    clustered = []
    current_cluster = [levels[0]]
    
    for i in range(1, len(levels)):
        if (levels[i] - current_cluster[-1]) / current_cluster[-1] < tolerance_pct:
            current_cluster.append(levels[i])
        else:
            # Save average of cluster
            clustered.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [levels[i]]
    
    if current_cluster:
        clustered.append(sum(current_cluster) / len(current_cluster))
    
    return clustered


@dataclass
class SRState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0
    resistance_levels: List[float] = None
    support_levels: List[float] = None


class SupportResistanceBreakout(Strategy):
    """
    Support/Resistance Breakout
    
    Entry Requirements:
    1. Identify S/R levels (pivot highs/lows)
    2. Price tests level 2+ times
    3. Breakout with volume (>1.5x avg)
    4. Strong breakout bar (close near high/low)
    5. Optional: Retest entry
    
    Stop: Behind S/R level
    Target: Next S/R level or 3R
    """
    
    def __init__(self, name: str = "sr_breakout", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, SRState] = {}
    
    def _get_state(self, symbol: str) -> SRState:
        if symbol not in self._state:
            self._state[symbol] = SRState()
            self._state[symbol].resistance_levels = []
            self._state[symbol].support_levels = []
        return self._state[symbol]
    
    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 60))
        entry_cutoff_minutes = int(self.params.get("entry_cutoff_minutes", 330))
        
        level_lookback = int(self.params.get("level_lookback", 100))  # Bars to find levels
        level_tolerance_pct = float(self.params.get("level_tolerance_pct", 0.005))  # 0.5% clustering
        
        min_touches = int(self.params.get("min_touches", 2))  # Level must be tested X times
        touch_tolerance_atr = float(self.params.get("touch_tolerance_atr", 0.3))
        
        atr_period = int(self.params.get("atr_period", 14))
        stop_buffer_atr = float(self.params.get("stop_buffer_atr", 0.5))
        risk_reward = float(self.params.get("risk_reward", 3.0))
        
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.5))
        
        min_breakout_strength = float(self.params.get("min_breakout_strength", 0.6))
        
        use_retest = bool(self.params.get("use_retest", False))
        retest_max_bars = int(self.params.get("retest_max_bars", 10))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))
        
        warmup = max(level_lookback, atr_period, volume_period) + 10
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
        
        if day_df.empty or len(day_df) < 20:
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
            # Recalculate levels each day
            st.resistance_levels = []
            st.support_levels = []
        
        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})
        
        # --- indicators ---
        atr = _atr(df, atr_period)
        vol_avg = df["volume"].astype(float).rolling(volume_period).mean()
        
        if pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})
        
        atr_now = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])
        
        if atr_now <= 0 or vol_avg_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})
        
        # --- identify S/R levels ---
        if not st.resistance_levels and not st.support_levels:
            recent_df = df.tail(level_lookback)
            all_levels = _find_pivot_levels(recent_df, level_lookback, level_tolerance_pct)
            
            if not all_levels:
                return Order(symbol, None, None, None, None, "no S/R levels found", {})
            
            current_price = _as_float(df["close"].iloc[-1])
            
            # Classify as support (below price) or resistance (above price)
            st.resistance_levels = [l for l in all_levels if l > current_price]
            st.support_levels = [l for l in all_levels if l < current_price]
            
            return Order(symbol, None, None, None, None, f"levels identified (R:{len(st.resistance_levels)}, S:{len(st.support_levels)})", {})
        
        # Current bar
        last = df.iloc[-1]
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        last_close = _as_float(last["close"])
        last_volume = _as_float(last["volume"])
        
        bar_range = last_high - last_low
        if bar_range <= 0:
            return Order(symbol, None, None, None, None, "zero range bar", {})
        
        # --- CHECK FOR BREAKOUTS ---
        touch_tolerance = touch_tolerance_atr * atr_now
        
        # Check resistance breakout (long)
        for resistance in st.resistance_levels:
            # Check if level was tested multiple times before
            touches = 0
            recent_df = df.tail(50)  # Check last 50 bars
            
            for _, bar in recent_df.iterrows():
                bar_high = _as_float(bar["high"])
                bar_low = _as_float(bar["low"])
                
                # Check if bar touched resistance
                if bar_high >= (resistance - touch_tolerance) and bar_low <= (resistance + touch_tolerance):
                    touches += 1
            
            if touches < min_touches:
                continue  # Not tested enough
            
            # Check if current bar broke above resistance
            if last_close > resistance and last_high > resistance:
                # Breakout confirmed!
                
                # Check volume
                if last_volume < (volume_mult * vol_avg_now):
                    continue
                
                # Check breakout strength (close in top 60% of bar)
                close_position = (last_close - last_low) / bar_range
                if close_position < min_breakout_strength:
                    continue
                
                # ENTRY
                entry = last_close
                stop = resistance - (stop_buffer_atr * atr_now)  # Stop below broken resistance
                
                # Target: Next resistance or 3R
                next_resistance = None
                for r in st.resistance_levels:
                    if r > entry:
                        next_resistance = r
                        break
                
                if next_resistance:
                    take = next_resistance
                else:
                    take = entry + risk_reward * (entry - stop)
                
                if stop >= entry or take <= entry:
                    continue
                
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="buy",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason=f"Resistance breakout at ${resistance:.2f} ({touches} tests)",
                    meta={
                        "resistance_level": resistance,
                        "touches": touches,
                        "volume_ratio": last_volume / vol_avg_now,
                        "atr": atr_now,
                        "trail_mode": self.params.get("trail_mode", "tiered"),
                        "trail_activate_after_partial": True,
                        "partial_take_pct": 0.5,
                        "partial_take_rr": 1.5,
                        "initial_stop": stop,
                    }
                )
        
        # Check support breakout (short)
        for support in st.support_levels:
            # Check if level was tested
            touches = 0
            recent_df = df.tail(50)
            
            for _, bar in recent_df.iterrows():
                bar_high = _as_float(bar["high"])
                bar_low = _as_float(bar["low"])
                
                if bar_low <= (support + touch_tolerance) and bar_high >= (support - touch_tolerance):
                    touches += 1
            
            if touches < min_touches:
                continue
            
            # Check if current bar broke below support
            if last_close < support and last_low < support:
                # Breakout confirmed!
                
                # Check volume
                if last_volume < (volume_mult * vol_avg_now):
                    continue
                
                # Check breakout strength (close in bottom 60%)
                close_position = (last_close - last_low) / bar_range
                if close_position > (1.0 - min_breakout_strength):
                    continue
                
                # ENTRY
                entry = last_close
                stop = support + (stop_buffer_atr * atr_now)
                
                # Target: Next support or 3R
                next_support = None
                for s in reversed(st.support_levels):
                    if s < entry:
                        next_support = s
                        break
                
                if next_support:
                    take = next_support
                else:
                    take = entry - risk_reward * (stop - entry)
                
                if stop <= entry or take >= entry:
                    continue
                
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="sell",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason=f"Support breakdown at ${support:.2f} ({touches} tests)",
                    meta={
                        "support_level": support,
                        "touches": touches,
                        "volume_ratio": last_volume / vol_avg_now,
                        "atr": atr_now,
                        "trail_mode": self.params.get("trail_mode", "tiered"),
                        "trail_activate_after_partial": True,
                        "partial_take_pct": 0.5,
                        "partial_take_rr": 1.5,
                        "initial_stop": stop,
                    }
                )
        
        return Order(symbol, None, None, None, None, "no S/R breakout", {})
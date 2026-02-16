"""
Bollinger Squeeze Breakout Strategy
Best for: All leveraged ETFs during consolidation (BOIL, UCO, TZA, TECL, SOXL)

Logic:
1. Bollinger Bands squeeze (low volatility)
2. Consolidation for 10+ bars
3. Breakout with volume expansion
4. Trade breakout direction
5. Target: 2x squeeze width

Why it works:
- Volatility cycles (compression → expansion)
- Breakouts from squeezes have high follow-through
- Volume confirms institutional participation
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


def _bollinger_bands(close: pd.Series, period: int, std_dev: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower


def _bb_width(upper: pd.Series, lower: pd.Series, sma: pd.Series) -> pd.Series:
    """Calculate BB width as percentage of middle band"""
    width = (upper - lower) / sma
    return width


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


@dataclass
class SqueezeState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class BollingerSqueeze(Strategy):
    """
    Bollinger Squeeze Breakout
    
    Entry Requirements:
    1. BB width in lowest 20th percentile (squeeze)
    2. Price consolidated 10+ bars
    3. Breakout above/below bands
    4. Volume >1.5x average
    5. Continuation bar (close near high/low)
    
    Stop: Inside squeeze range
    Target: 2x squeeze width
    """
    
    def __init__(self, name: str = "bollinger_squeeze", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, SqueezeState] = {}
    
    def _get_state(self, symbol: str) -> SqueezeState:
        if symbol not in self._state:
            self._state[symbol] = SqueezeState()
        return self._state[symbol]
    
    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 30))
        entry_cutoff_minutes = int(self.params.get("entry_cutoff_minutes", 330))
        
        bb_period = int(self.params.get("bb_period", 20))
        bb_std_dev = float(self.params.get("bb_std_dev", 2.0))
        
        squeeze_lookback = int(self.params.get("squeeze_lookback", 100))  # Bars to check percentile
        squeeze_percentile = float(self.params.get("squeeze_percentile", 20))  # Must be in lowest 20%
        
        min_consolidation_bars = int(self.params.get("min_consolidation_bars", 10))
        
        atr_period = int(self.params.get("atr_period", 14))
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.0))
        target_squeeze_mult = float(self.params.get("target_squeeze_mult", 2.0))
        
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.5))
        
        min_breakout_strength = float(self.params.get("min_breakout_strength", 0.6))  # Close in top/bottom 60% of bar
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))
        
        warmup = max(bb_period, atr_period, volume_period, squeeze_lookback) + 5
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
        
        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})
        
        # --- indicators ---
        close = df["close"].astype(float)
        upper, middle, lower = _bollinger_bands(close, bb_period, bb_std_dev)
        bb_width_series = _bb_width(upper, lower, middle)
        
        atr = _atr(df, atr_period)
        vol_avg = df["volume"].astype(float).rolling(volume_period).mean()
        
        if pd.isna(upper.iloc[-1]) or pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})
        
        # Current values
        upper_now = _as_float(upper.iloc[-1])
        middle_now = _as_float(middle.iloc[-1])
        lower_now = _as_float(lower.iloc[-1])
        bb_width_now = _as_float(bb_width_series.iloc[-1])
        atr_now = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])
        
        if atr_now <= 0 or vol_avg_now <= 0 or bb_width_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})
        
        # Current bar
        last = df.iloc[-1]
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        last_close = _as_float(last["close"])
        last_volume = _as_float(last["volume"])
        
        # --- CHECK 1: Is BB width in squeeze? (lowest X percentile) ---
        recent_widths = bb_width_series.tail(squeeze_lookback).dropna()
        if len(recent_widths) < squeeze_lookback:
            return Order(symbol, None, None, None, None, "not enough width history", {})
        
        width_threshold = recent_widths.quantile(squeeze_percentile / 100.0)
        
        if bb_width_now > width_threshold:
            return Order(symbol, None, None, None, None, f"not in squeeze (width {bb_width_now:.4f} > {width_threshold:.4f})", {})
        
        # --- CHECK 2: Has price consolidated? ---
        recent_df = df.tail(min_consolidation_bars + 1)
        consolidation_range = recent_df["high"].max() - recent_df["low"].min()
        consolidation_range_pct = consolidation_range / middle_now
        
        # Consolidation should be tight (< 2% range)
        if consolidation_range_pct > 0.02:
            return Order(symbol, None, None, None, None, f"not consolidated (range {consolidation_range_pct*100:.1f}%)", {})
        
        # --- CHECK 3: Volume expansion ---
        if last_volume < (volume_mult * vol_avg_now):
            return Order(symbol, None, None, None, None, "insufficient volume", {})
        
        # --- CHECK 4: Breakout strength (close near high/low) ---
        bar_range = last_high - last_low
        if bar_range <= 0:
            return Order(symbol, None, None, None, None, "zero range bar", {})
        
        # For bullish: close should be in top 60% of bar
        # For bearish: close should be in bottom 60% of bar
        close_position = (last_close - last_low) / bar_range  # 0 = low, 1 = high
        
        # --- LONG BREAKOUT ---
        if last_close > upper_now:
            # Bullish breakout
            if close_position < (1.0 - min_breakout_strength):
                return Order(symbol, None, None, None, None, "weak bullish close", {})
            
            # Calculate squeeze range
            squeeze_high = recent_df["high"].max()
            squeeze_low = recent_df["low"].min()
            squeeze_width = squeeze_high - squeeze_low
            
            entry = last_close
            stop = squeeze_low - (stop_atr_mult * atr_now)  # Stop below squeeze
            take = entry + (target_squeeze_mult * squeeze_width)  # Target: 2x squeeze width
            
            if stop >= entry or take <= entry:
                return Order(symbol, None, None, None, None, "invalid long levels", {})
            
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"Bollinger squeeze breakout long (width {bb_width_now:.4f})",
                meta={
                    "bb_width": bb_width_now,
                    "squeeze_width": squeeze_width,
                    "consolidation_bars": min_consolidation_bars,
                    "volume_ratio": last_volume / vol_avg_now,
                    "atr": atr_now,
                    "trail_mode": self.params.get("trail_mode", "tiered"),
                    "trail_activate_after_partial": True,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 1.5,
                    "initial_stop": stop,
                }
            )
        
        # --- SHORT BREAKOUT ---
        elif last_close < lower_now:
            # Bearish breakout
            if close_position > min_breakout_strength:
                return Order(symbol, None, None, None, None, "weak bearish close", {})
            
            # Calculate squeeze range
            squeeze_high = recent_df["high"].max()
            squeeze_low = recent_df["low"].min()
            squeeze_width = squeeze_high - squeeze_low
            
            entry = last_close
            stop = squeeze_high + (stop_atr_mult * atr_now)  # Stop above squeeze
            take = entry - (target_squeeze_mult * squeeze_width)  # Target: 2x squeeze width
            
            if stop <= entry or take >= entry:
                return Order(symbol, None, None, None, None, "invalid short levels", {})
            
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"Bollinger squeeze breakout short (width {bb_width_now:.4f})",
                meta={
                    "bb_width": bb_width_now,
                    "squeeze_width": squeeze_width,
                    "consolidation_bars": min_consolidation_bars,
                    "volume_ratio": last_volume / vol_avg_now,
                    "atr": atr_now,
                    "trail_mode": self.params.get("trail_mode", "tiered"),
                    "trail_activate_after_partial": True,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 1.5,
                    "initial_stop": stop,
                }
            )
        
        return Order(symbol, None, None, None, None, "no squeeze breakout", {})
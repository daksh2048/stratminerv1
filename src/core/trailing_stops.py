"""
Advanced Trailing Stop Module

Provides multiple trailing stop strategies that can be configured per-position
via the position.meta dict.

Usage in strategy:
    meta = {
        "trail_mode": "atr",           # or "chandelier", "tiered", "hybrid"
        "trail_atr_mult": 2.0,         # for ATR-based modes
        "trail_lookback": 10,          # for chandelier
        "trail_min_pct": 0.01,         # minimum trail (safety)
        "trail_max_pct": 0.05,         # maximum trail (safety)
    }
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


def _as_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    if isinstance(x, np.generic):
        return float(x)
    return float(x)


def calculate_trailing_stop(
    pos_side: str,
    pos_entry: float,
    current_stop: float,
    candle_high: float,
    candle_low: float,
    meta: dict,
    df_recent: Optional[pd.DataFrame] = None,  # Last N candles for structure/chandelier
    atr_current: Optional[float] = None,
) -> float:
    """
    Calculate new trailing stop based on configured mode.
    
    Args:
        pos_side: "buy" or "sell"
        pos_entry: Entry price
        current_stop: Current stop price
        candle_high: Current candle high
        candle_low: Current candle low
        meta: Position metadata dict with trail config
        df_recent: Recent candles DataFrame (for structure/chandelier modes)
        atr_current: Current ATR value (for ATR-based modes)
    
    Returns:
        New stop price (will never be worse than current_stop)
    """
    
    mode = str(meta.get("trail_mode", "percent")).lower()
    
    # Update peak/trough
    if pos_side == "buy":
        current_peak = float(meta.get("peak", pos_entry))
        new_peak = max(current_peak, candle_high)
        meta["peak"] = new_peak
    else:
        current_trough = float(meta.get("trough", pos_entry))
        new_trough = min(current_trough, candle_low)
        meta["trough"] = new_trough
    
    # Calculate new stop based on mode
    if mode == "percent":
        new_stop = _trail_percent(pos_side, meta)
    
    elif mode == "atr":
        new_stop = _trail_atr(pos_side, meta, atr_current)
    
    elif mode == "chandelier":
        new_stop = _trail_chandelier(pos_side, meta, df_recent, atr_current)
    
    elif mode == "tiered":
        new_stop = _trail_tiered(pos_side, pos_entry, meta)
    
    elif mode == "structure":
        new_stop = _trail_structure(pos_side, meta, df_recent, atr_current)
    
    elif mode == "hybrid":
        new_stop = _trail_hybrid(pos_side, meta, df_recent, atr_current)
    
    elif mode == "parabolic":
        new_stop = _trail_parabolic(pos_side, meta)
    
    else:
        # Fallback to percent
        new_stop = _trail_percent(pos_side, meta)
    
    # Safety: never move stop against position
    if pos_side == "buy":
        return max(current_stop, new_stop) if new_stop is not None else current_stop
    else:
        return min(current_stop, new_stop) if new_stop is not None else current_stop


# ============================================================================
# TRAILING STOP STRATEGIES
# ============================================================================

def _trail_percent(pos_side: str, meta: dict) -> Optional[float]:
    """Simple percentage-based trailing (original method)"""
    trail_pct = float(meta.get("trail_pct", 0.02))
    
    if pos_side == "buy":
        peak = float(meta.get("peak", 0))
        return peak * (1.0 - trail_pct)
    else:
        trough = float(meta.get("trough", 0))
        return trough * (1.0 + trail_pct)


def _trail_atr(pos_side: str, meta: dict, atr: Optional[float]) -> Optional[float]:
    """ATR-based trailing - adapts to volatility"""
    if atr is None or atr <= 0:
        return _trail_percent(pos_side, meta)  # Fallback
    
    trail_atr_mult = float(meta.get("trail_atr_mult", 2.0))
    
    if pos_side == "buy":
        peak = float(meta.get("peak", 0))
        return peak - (trail_atr_mult * atr)
    else:
        trough = float(meta.get("trough", 0))
        return trough + (trail_atr_mult * atr)


def _trail_chandelier(
    pos_side: str,
    meta: dict,
    df_recent: Optional[pd.DataFrame],
    atr: Optional[float]
) -> Optional[float]:
    """Chandelier stop - trail based on highest high / lowest low"""
    if df_recent is None or df_recent.empty or atr is None or atr <= 0:
        return _trail_atr(pos_side, meta, atr)  # Fallback
    
    lookback = int(meta.get("trail_lookback", 10))
    trail_atr_mult = float(meta.get("trail_atr_mult", 2.0))
    
    if len(df_recent) < lookback:
        return _trail_atr(pos_side, meta, atr)
    
    if pos_side == "buy":
        highest_high = _as_float(df_recent["high"].tail(lookback).max())
        return highest_high - (trail_atr_mult * atr)
    else:
        lowest_low = _as_float(df_recent["low"].tail(lookback).min())
        return lowest_low + (trail_atr_mult * atr)


def _trail_tiered(pos_side: str, pos_entry: float, meta: dict) -> Optional[float]:
    """Profit-tiered trailing - tighter as profit grows"""
    initial_stop = float(meta.get("initial_stop", 0))
    initial_risk = abs(pos_entry - initial_stop)
    
    if initial_risk <= 0:
        return _trail_percent(pos_side, meta)
    
    if pos_side == "buy":
        peak = float(meta.get("peak", pos_entry))
        current_profit = peak - pos_entry
    else:
        trough = float(meta.get("trough", pos_entry))
        current_profit = pos_entry - trough
    
    r_multiple = current_profit / initial_risk
    
    # Tiered percentages
    if r_multiple < 1.0:
        trail_pct = float(meta.get("trail_tier1_pct", 0.04))  # 4% - loose
    elif r_multiple < 2.0:
        trail_pct = float(meta.get("trail_tier2_pct", 0.02))  # 2% - medium
    else:
        trail_pct = float(meta.get("trail_tier3_pct", 0.01))  # 1% - tight
    
    if pos_side == "buy":
        peak = float(meta.get("peak", 0))
        return peak * (1.0 - trail_pct)
    else:
        trough = float(meta.get("trough", 0))
        return trough * (1.0 + trail_pct)


def _trail_structure(
    pos_side: str,
    meta: dict,
    df_recent: Optional[pd.DataFrame],
    atr: Optional[float]
) -> Optional[float]:
    """Structure-based trailing - respects swing points"""
    if df_recent is None or df_recent.empty or atr is None:
        return _trail_percent(pos_side, meta)
    
    swing_window = int(meta.get("trail_swing_window", 5))
    buffer_atr_mult = float(meta.get("trail_buffer_mult", 0.3))
    
    if len(df_recent) < swing_window * 2 + 1:
        return _trail_percent(pos_side, meta)
    
    # Detect recent swing
    if pos_side == "buy":
        # Find last swing low
        lows = df_recent["low"].tail(swing_window * 2).values
        swing_idx = None
        for i in range(swing_window, len(lows) - swing_window):
            left = lows[i-swing_window:i]
            right = lows[i+1:i+swing_window+1]
            if lows[i] < min(left.min(), right.min()):
                swing_idx = i
        
        if swing_idx is not None:
            swing_low = float(lows[swing_idx])
            return swing_low - (buffer_atr_mult * atr)
    else:
        # Find last swing high
        highs = df_recent["high"].tail(swing_window * 2).values
        swing_idx = None
        for i in range(swing_window, len(highs) - swing_window):
            left = highs[i-swing_window:i]
            right = highs[i+1:i+swing_window+1]
            if highs[i] > max(left.max(), right.max()):
                swing_idx = i
        
        if swing_idx is not None:
            swing_high = float(highs[swing_idx])
            return swing_high + (buffer_atr_mult * atr)
    
    # Fallback if no swing found
    return _trail_atr(pos_side, meta, atr)


def _trail_hybrid(
    pos_side: str,
    meta: dict,
    df_recent: Optional[pd.DataFrame],
    atr: Optional[float]
) -> Optional[float]:
    """Hybrid - combines ATR + structure, uses the more conservative"""
    atr_stop = _trail_atr(pos_side, meta, atr)
    structure_stop = _trail_structure(pos_side, meta, df_recent, atr)
    
    if atr_stop is None:
        return structure_stop
    if structure_stop is None:
        return atr_stop
    
    # Use the stop that's further from peak/trough (more conservative)
    if pos_side == "buy":
        return max(atr_stop, structure_stop)
    else:
        return min(atr_stop, structure_stop)


def _trail_parabolic(pos_side: str, meta: dict) -> Optional[float]:
    """Parabolic trailing - accelerates over time"""
    initial_trail = float(meta.get("trail_initial_pct", 0.03))  # 3% start
    acceleration = float(meta.get("trail_acceleration", 0.0005))  # 0.05% per bar
    min_trail = float(meta.get("trail_min_pct", 0.01))  # 1% minimum
    
    # Track bars since entry
    bars_in_position = int(meta.get("bars_in_position", 0))
    meta["bars_in_position"] = bars_in_position + 1
    
    # Calculate current trail percentage (decreases over time)
    current_trail_pct = max(min_trail, initial_trail - (acceleration * bars_in_position))
    
    if pos_side == "buy":
        peak = float(meta.get("peak", 0))
        return peak * (1.0 - current_trail_pct)
    else:
        trough = float(meta.get("trough", 0))
        return trough * (1.0 + current_trail_pct)


# ============================================================================
# HELPER: Calculate ATR from recent candles
# ============================================================================

def calculate_atr(df_recent: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calculate ATR from recent candles"""
    if df_recent is None or len(df_recent) < period:
        return None
    
    high = df_recent["high"].astype(float)
    low = df_recent["low"].astype(float)
    close = df_recent["close"].astype(float)
    
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean().iloc[-1]
    
    if pd.isna(atr):
        return None
    
    return float(atr)
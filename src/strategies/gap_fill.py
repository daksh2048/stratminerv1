"""
Gap Fill Strategy - Mean Reversion
Best for: Volatile commodity ETFs (BOIL, KOLD, UCO, UGL)

Logic:
1. Gap at open (>0.8 ATR from previous close)
2. Price fails to fill gap in first 30 minutes
3. Trade against the gap (fade the gap)
4. Target: Fill the gap
5. Stop: Beyond gap extreme

Why it works:
- Gaps create inefficiency
- Market tends to fill gaps 60-70% of time
- Works best on mean-reverting instruments
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


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


@dataclass
class GapState:
    day: Optional[pd.Timestamp] = None
    gap_identified: bool = False
    gap_direction: Optional[str] = None  # "up" or "down"
    gap_size: float = 0.0
    gap_price: float = 0.0  # Previous close (gap fill target)
    trade_taken: bool = False


class GapFill(Strategy):
    """
    Gap Fill Mean Reversion
    
    Entry Requirements:
    1. Gap >0.8 ATR from previous close
    2. 30 minutes after open (gap hasn't filled)
    3. Volume confirmation (>1.2x average)
    4. Trade towards gap fill
    
    Stop: Beyond gap extreme + buffer
    Target: Previous close (gap fill)
    """
    
    def __init__(self, name: str = "gap_fill", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, GapState] = {}
    
    def _get_state(self, symbol: str) -> GapState:
        if symbol not in self._state:
            self._state[symbol] = GapState()
        return self._state[symbol]
    
    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_gap_atr = float(self.params.get("min_gap_atr", 0.8))  # Minimum gap size
        gap_window_minutes = int(self.params.get("gap_window_minutes", 30))  # Time to wait before fading
        
        atr_period = int(self.params.get("atr_period", 14))
        stop_buffer_atr = float(self.params.get("stop_buffer_atr", 0.5))
        
        volume_period = int(self.params.get("volume_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.2))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 1))
        
        warmup = max(atr_period, volume_period) + 5
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
        
        if day_df.empty or len(day_df) < 10:
            return Order(symbol, None, None, None, None, "not enough bars today", {})
        
        # --- time gates ---
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        # --- state management ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.gap_identified = False
            st.gap_direction = None
            st.gap_size = 0.0
            st.gap_price = 0.0
            st.trade_taken = False
        
        if st.trade_taken:
            return Order(symbol, None, None, None, None, "already traded gap today", {})
        
        # --- indicators ---
        atr = _atr(df, atr_period)
        vol_avg = df["volume"].astype(float).rolling(volume_period).mean()
        
        if pd.isna(atr.iloc[-1]) or pd.isna(vol_avg.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})
        
        atr_now = _as_float(atr.iloc[-1])
        vol_avg_now = _as_float(vol_avg.iloc[-1])
        
        if atr_now <= 0 or vol_avg_now <= 0:
            return Order(symbol, None, None, None, None, "invalid indicators", {})
        
        # --- identify gap at open ---
        if not st.gap_identified:
            # Need at least a few bars to identify gap
            if len(day_df) < 3:
                return Order(symbol, None, None, None, None, "waiting for gap identification", {})
            
            # Get yesterday's close
            yesterday_df = df.loc[~in_today].copy()
            if yesterday_df.empty:
                return Order(symbol, None, None, None, None, "no yesterday data", {})
            
            yesterday_close = _as_float(yesterday_df["close"].iloc[-1])
            
            # Get today's open
            today_open = _as_float(day_df["open"].iloc[0])
            
            # Calculate gap
            gap_size = abs(today_open - yesterday_close)
            gap_size_atr = gap_size / atr_now
            
            if gap_size_atr >= min_gap_atr:
                # Gap identified!
                st.gap_identified = True
                st.gap_size = gap_size
                st.gap_price = yesterday_close
                
                if today_open > yesterday_close:
                    st.gap_direction = "up"  # Gap up
                else:
                    st.gap_direction = "down"  # Gap down
                
                print(f"[GAP] {symbol} gap {st.gap_direction} {gap_size_atr:.2f} ATR | Target: ${yesterday_close:.2f}")
            
            return Order(symbol, None, None, None, None, "gap identification phase", {})
        
        # --- wait for gap window ---
        mins_from_open = (last_ts_m - session_open).total_seconds() / 60.0
        if mins_from_open < gap_window_minutes:
            return Order(symbol, None, None, None, None, f"waiting {gap_window_minutes} min", {})
        
        # --- check if gap already filled ---
        if st.gap_direction == "up":
            # Gap up - check if price dropped to yesterday's close
            if day_df["low"].min() <= st.gap_price:
                return Order(symbol, None, None, None, None, "gap already filled", {})
        else:
            # Gap down - check if price rose to yesterday's close
            if day_df["high"].max() >= st.gap_price:
                return Order(symbol, None, None, None, None, "gap already filled", {})
        
        # --- volume confirmation ---
        last = day_df.iloc[-1]
        last_close = _as_float(last["close"])
        last_volume = _as_float(last["volume"])
        
        if last_volume < (volume_mult * vol_avg_now):
            return Order(symbol, None, None, None, None, "insufficient volume", {})
        
        # --- ENTRY: Fade the gap ---
        if st.gap_direction == "up":
            # Gap up → Go SHORT (fade towards gap fill)
            entry = last_close
            stop = day_df["high"].max() + (stop_buffer_atr * atr_now)  # Stop above high of day
            take = st.gap_price  # Target: Fill the gap
            
            if stop <= entry or take >= entry:
                return Order(symbol, None, None, None, None, "invalid short levels", {})
            
            st.trade_taken = True
            
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"Gap fill short (gap {st.gap_size/atr_now:.2f} ATR)",
                meta={
                    "gap_size_atr": st.gap_size / atr_now,
                    "gap_target": st.gap_price,
                    "atr": atr_now,
                    "volume_ratio": last_volume / vol_avg_now,
                    "trail_mode": self.params.get("trail_mode", "percent"),
                    "trail_pct": 0.015,  # Tight trail for MR
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 0.8,  # Take half at 80% of gap fill
                    "initial_stop": stop,
                }
            )
        
        elif st.gap_direction == "down":
            # Gap down → Go LONG (fade towards gap fill)
            entry = last_close
            stop = day_df["low"].min() - (stop_buffer_atr * atr_now)  # Stop below low of day
            take = st.gap_price  # Target: Fill the gap
            
            if stop >= entry or take <= entry:
                return Order(symbol, None, None, None, None, "invalid long levels", {})
            
            st.trade_taken = True
            
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason=f"Gap fill long (gap {st.gap_size/atr_now:.2f} ATR)",
                meta={
                    "gap_size_atr": st.gap_size / atr_now,
                    "gap_target": st.gap_price,
                    "atr": atr_now,
                    "volume_ratio": last_volume / vol_avg_now,
                    "trail_mode": self.params.get("trail_mode", "percent"),
                    "trail_pct": 0.015,
                    "partial_take_pct": 0.5,
                    "partial_take_rr": 0.8,
                    "initial_stop": stop,
                }
            )
        
        return Order(symbol, None, None, None, None, "no gap setup", {})
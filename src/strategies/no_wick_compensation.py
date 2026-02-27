"""
No-Wick / Compensation Play Strategy
Based on YouTuber's "90% win rate" system

Entry Logic:
1. Identify no-wick candles (bullish: no bottom wick, bearish: no top wick)
2. Must be WITH trend (higher highs/lows for bullish, lower highs/lows for bearish)
3. Wait for price to retrace and tap the no-wick candle
4. Maximum 9 candles between creation and tap
5. Enter with trend
6. Stop: Most recent higher low (bullish) or lower high (bearish) + breathing room
7. TP: 1:1 risk/reward

Filters:
- No imbalances against position
- Avoid first 3 hours of Asia session
- Don't hold through NY close
- Max 9 candles between signal and tap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

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


@dataclass
class NoWickCandle:
    """Store no-wick candle information"""
    index: int
    price: float  # Open/close price of no-wick
    side: str  # 'buy' or 'sell'
    candle_time: pd.Timestamp


@dataclass
class NoWickState:
    """Track state for No-Wick strategy"""
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0
    pending_signals: List[NoWickCandle] = None
    
    def __post_init__(self):
        if self.pending_signals is None:
            self.pending_signals = []


class NoWickCompensationPlay(Strategy):
    """
    No-Wick Compensation Play Strategy
    
    Rules from video:
    - 15-minute timeframe
    - Trade with trend only
    - No-wick candles must tap within 9 candles
    - Breathing room: AUDUSD 3.8 pips, GBPUSD 4 pips, USDJPY 5.5 pips
    - 1:1 risk/reward
    - Avoid first 3 hours Asia, don't hold through NY close
    """
    
    def __init__(self, name: str = "no_wick", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, NoWickState] = {}
    
    def _get_state(self, symbol: str) -> NoWickState:
        if symbol not in self._state:
            self._state[symbol] = NoWickState()
        return self._state[symbol]
    
    def _is_no_wick_bullish(self, row) -> bool:
        """Check if candle is bullish with no bottom wick"""
        open_price = _as_float(row['open'])
        low_price = _as_float(row['low'])
        close_price = _as_float(row['close'])
        
        # Bullish candle
        if close_price <= open_price:
            return False
        
        # No bottom wick (or minimal - within 0.1% tolerance)
        body_start = min(open_price, close_price)
        wick_size = body_start - low_price
        body_size = abs(close_price - open_price)
        
        if body_size == 0:
            return False
        
        wick_ratio = wick_size / body_size
        
        # Allow tiny wick (< 5% of body)
        return wick_ratio < 0.05
    
    def _is_no_wick_bearish(self, row) -> bool:
        """Check if candle is bearish with no top wick"""
        open_price = _as_float(row['open'])
        high_price = _as_float(row['high'])
        close_price = _as_float(row['close'])
        
        # Bearish candle
        if close_price >= open_price:
            return False
        
        # No top wick (or minimal - within 0.1% tolerance)
        body_start = max(open_price, close_price)
        wick_size = high_price - body_start
        body_size = abs(close_price - open_price)
        
        if body_size == 0:
            return False
        
        wick_ratio = wick_size / body_size
        
        # Allow tiny wick (< 5% of body)
        return wick_ratio < 0.05
    
    def _identify_trend(self, df: pd.DataFrame, lookback: int = 20) -> str:
        """
        Identify trend based on higher highs/higher lows
        
        Returns: 'bullish', 'bearish', or 'none'
        """
        if len(df) < lookback:
            return 'none'
        
        closes = df['close'].values[-lookback:]
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        
        # Find pivots (simplified - look for local highs/lows)
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(closes) - 2):
            # Pivot high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivot_highs.append((i, highs[i]))
            
            # Pivot low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivot_lows.append((i, lows[i]))
        
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return 'none'
        
        # Check for higher highs and higher lows (bullish)
        recent_highs = pivot_highs[-2:]
        recent_lows = pivot_lows[-2:]
        
        if recent_highs[1][1] > recent_highs[0][1] and \
           recent_lows[1][1] > recent_lows[0][1]:
            return 'bullish'
        
        # Check for lower highs and lower lows (bearish)
        if recent_highs[1][1] < recent_highs[0][1] and \
           recent_lows[1][1] < recent_lows[0][1]:
            return 'bearish'
        
        return 'none'
    
    def _find_recent_structure(self, df: pd.DataFrame, side: str) -> Tuple[float, int]:
        """
        Find most recent higher low (for buys) or lower high (for sells)
        
        Returns: (price, index)
        """
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if side == 'buy':
            # Find most recent higher low
            for i in range(len(closes) - 3, 1, -1):
                # Check if this is a pivot low
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    return lows[i], i
        else:
            # Find most recent lower high
            for i in range(len(closes) - 3, 1, -1):
                # Check if this is a pivot high
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    return highs[i], i
        
        # Fallback
        if side == 'buy':
            recent_low = np.min(lows[-20:])
            return recent_low, len(lows) - 20
        else:
            recent_high = np.max(highs[-20:])
            return recent_high, len(highs) - 20
    
    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        
        # Breathing room by pair (in pips)
        pair = symbol.replace("/", "")
        breathing_room_pips = {
            'AUDUSD': 3.8,
            'GBPUSD': 4.0,
            'USDJPY': 5.5,
            'AUDJPY': 5.0
        }.get(pair, 4.0)
        
        # Convert pips to price (for JPY pairs, 1 pip = 0.01, others = 0.0001)
        if 'JPY' in pair:
            breathing_room = breathing_room_pips * 0.01
        else:
            breathing_room = breathing_room_pips * 0.0001
        
        max_candles_to_tap = int(self.params.get("max_candles_to_tap", 9))
        max_trades_per_day = int(self.params.get("max_trades_per_day", 5))
        trend_lookback = int(self.params.get("trend_lookback", 20))
        
        warmup = 50
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})
        
        # --- timezone ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        # --- state ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
            st.pending_signals = []
        
        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})
        
        # --- identify trend ---
        trend = self._identify_trend(df, trend_lookback)
        
        if trend == 'none':
            return Order(symbol, None, None, None, None, "no clear trend", {})
        
        # --- check for new no-wick candles ---
        current_row = df.iloc[-1]
        
        # Bullish no-wick in bullish trend
        if trend == 'bullish' and self._is_no_wick_bullish(current_row):
            signal = NoWickCandle(
                index=len(df) - 1,
                price=_as_float(current_row['open']),  # Use open as entry level
                side='buy',
                candle_time=last_ts_m
            )
            st.pending_signals.append(signal)
        
        # Bearish no-wick in bearish trend
        if trend == 'bearish' and self._is_no_wick_bearish(current_row):
            signal = NoWickCandle(
                index=len(df) - 1,
                price=_as_float(current_row['open']),  # Use open as entry level
                side='sell',
                candle_time=last_ts_m
            )
            st.pending_signals.append(signal)
        
        # --- check if any pending signals have been tapped ---
        current_low = _as_float(current_row['low'])
        current_high = _as_float(current_row['high'])
        current_index = len(df) - 1
        
        for signal in st.pending_signals:
            candles_elapsed = current_index - signal.index
            
            # Remove signal if too old (>9 candles)
            if candles_elapsed > max_candles_to_tap:
                st.pending_signals.remove(signal)
                continue
            
            # Don't check on same candle as signal
            if candles_elapsed == 0:
                continue
            
            # Check if tapped
            tapped = False
            
            if signal.side == 'buy' and current_low <= signal.price:
                tapped = True
            elif signal.side == 'sell' and current_high >= signal.price:
                tapped = True
            
            if not tapped:
                continue
            
            # ENTRY CONDITIONS MET
            entry = signal.price
            
            # Find stop loss (most recent structure + breathing room)
            stop_price, stop_index = self._find_recent_structure(df, signal.side)
            
            if signal.side == 'buy':
                stop = stop_price - breathing_room
                
                if stop >= entry:
                    continue  # Invalid stop
                
                # TP at 1:1
                risk = entry - stop
                take = entry + risk
            else:
                stop = stop_price + breathing_room
                
                if stop <= entry:
                    continue  # Invalid stop
                
                # TP at 1:1
                risk = stop - entry
                take = entry - risk
            
            # Remove this signal
            st.pending_signals.remove(signal)
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side=signal.side,
                entry=entry,
                stop=stop,
                take=take,
                reason=f"No-wick {signal.side} tap after {candles_elapsed} candles",
                meta={
                    "candles_to_tap": candles_elapsed,
                    "breathing_room": breathing_room,
                    "trend": trend,
                    "trail_mode": "fixed",  # No trailing for this strategy
                }
            )
        
        return Order(symbol, None, None, None, None, "waiting for tap", {})
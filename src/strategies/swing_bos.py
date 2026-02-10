from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

from src.core.types import Order
from src.core.data import CandleFeed
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


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _detect_swing_highs(high: pd.Series, window: int) -> pd.Series:
    """
    Detect swing highs: bar where high > all bars within window on both sides.
    Returns boolean Series.
    """
    swings = pd.Series(False, index=high.index)
    for i in range(window, len(high) - window):
        current = high.iloc[i]
        left = high.iloc[i-window:i]
        right = high.iloc[i+1:i+window+1]
        if current > left.max() and current > right.max():
            swings.iloc[i] = True
    return swings


def _detect_swing_lows(low: pd.Series, window: int) -> pd.Series:
    """
    Detect swing lows: bar where low < all bars within window on both sides.
    Returns boolean Series.
    """
    swings = pd.Series(False, index=low.index)
    for i in range(window, len(low) - window):
        current = low.iloc[i]
        left = low.iloc[i-window:i]
        right = low.iloc[i+1:i+window+1]
        if current < left.min() and current < right.min():
            swings.iloc[i] = True
    return swings


@dataclass
class BOSState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    htf_cache: Optional[pd.DataFrame] = None
    htf_last_update: Optional[pd.Timestamp] = None


class SwingBOS(Strategy):
    """
    Swing Break of Structure (BOS) Strategy
    
    Multi-Timeframe Approach:
    1. HTF Bias (1h default): Determines overall trend using EMA
       - Bullish: Price above HTF EMA → only take LONG setups
       - Bearish: Price below HTF EMA → only take SHORT setups
    
    2. LTF Entry (5m default): Detects swing structure breaks
       - LONG: Price breaks above previous swing high + bullish HTF
       - SHORT: Price breaks below previous swing low + bearish HTF
    
    3. Risk Management:
       - Stop: ATR-based distance from swing point
       - Take: RR-based (default 2.5:1)
       - Trailing stop: Activated from entry (2% default)
    
    4. Filters:
       - Minimum ATR (volatility filter)
       - RTH hours only
       - Max trades per day
    
    Parameters:
    - htf_timeframe: Higher timeframe for bias (default: "1h")
    - htf_ema_period: EMA period on HTF (default: 20)
    - swing_window: Bars on each side for swing detection (default: 5)
    - atr_period: ATR calculation period (default: 14)
    - stop_atr_mult: ATR multiplier for stop distance (default: 1.5)
    - risk_reward: RR ratio (default: 2.5)
    - min_atr: Minimum ATR filter (default: 0.0)
    - trail_pct: Trailing stop percentage (default: 0.02 = 2%)
    - max_trades_per_day: Max trades per symbol per day (default: 2)
    """

    def __init__(self, name: str = "swing_bos", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, BOSState] = {}

    def _get_state(self, symbol: str) -> BOSState:
        if symbol not in self._state:
            self._state[symbol] = BOSState()
        return self._state[symbol]

    def _fetch_htf_bias(
        self,
        symbol: str,
        ltf_last_ts: pd.Timestamp,
        market_tz: str
    ) -> Optional[str]:
        """
        Fetch HTF data and determine trend bias.
        Returns: 'bullish', 'bearish', or None
        
        Caches HTF data for 1 hour to avoid repeated fetches.
        """
        st = self._get_state(symbol)
        htf_timeframe = str(self.params.get("htf_timeframe", "1h"))
        
        # Check cache validity (1 hour)
        if st.htf_cache is not None and st.htf_last_update is not None:
            cache_age_hours = (ltf_last_ts - st.htf_last_update).total_seconds() / 3600
            if cache_age_hours < 1.0:
                htf_df = st.htf_cache
            else:
                # Re-fetch HTF data
                try:
                    feed = CandleFeed(exchange="yahoo", symbol=symbol, timeframe=htf_timeframe)
                    htf_df = feed.fetch(period="60d", limit=100)
                    st.htf_cache = htf_df
                    st.htf_last_update = ltf_last_ts
                except Exception as e:
                    return None
        else:
            # First fetch
            try:
                feed = CandleFeed(exchange="yahoo", symbol=symbol, timeframe=htf_timeframe)
                htf_df = feed.fetch(period="60d", limit=100)
                st.htf_cache = htf_df
                st.htf_last_update = ltf_last_ts
            except Exception as e:
                return None

        if htf_df is None or htf_df.empty:
            return None

        htf_ema_period = int(self.params.get("htf_ema_period", 20))
        
        # Calculate HTF EMA
        htf_ema = _ema(htf_df["close"].astype(float), htf_ema_period)
        
        if pd.isna(htf_ema.iloc[-1]):
            return None
        
        last_close = _as_float(htf_df["close"].iloc[-1])
        last_ema = _as_float(htf_ema.iloc[-1])
        
        # Determine bias
        if last_close > last_ema:
            return "bullish"
        elif last_close < last_ema:
            return "bearish"
        
        return None

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- Parameters ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        swing_window = int(self.params.get("swing_window", 5))
        atr_period = int(self.params.get("atr_period", 14))
        
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.5))
        risk_reward = float(self.params.get("risk_reward", 2.5))
        trail_pct = float(self.params.get("trail_pct", 0.02))  # 2% trailing
        
        min_atr = float(self.params.get("min_atr", 0.0))
        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 30))
        entry_cutoff_minutes = int(self.params.get("entry_cutoff_minutes", 240))

        # Warmup requirement
        warmup = max(atr_period + swing_window * 2, 50)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        # --- Timezone conversion ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        # Session bounds
        oh, om = map(int, rth_open.split(":"))
        ch, cm = map(int, rth_close.split(":"))
        session_open = day_m + pd.Timedelta(hours=oh, minutes=om)
        session_close = day_m + pd.Timedelta(hours=ch, minutes=cm)

        # RTH check
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        # Time window gates
        mins_from_open = (last_ts_m - session_open).total_seconds() / 60.0
        if mins_from_open < min_minutes_after_open:
            return Order(symbol, None, None, None, None, "too early", {"mins_from_open": mins_from_open})
        if mins_from_open > entry_cutoff_minutes:
            return Order(symbol, None, None, None, None, "past cutoff", {"mins_from_open": mins_from_open})

        # --- State management ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
            # Reset swing memory at day start
            st.last_swing_high = None
            st.last_swing_low = None

        if st.trades_today >= max_trades_per_day:
            return Order(
                symbol, None, None, None, None,
                "max trades/day",
                {"trades_today": st.trades_today}
            )

        # --- HTF Bias ---
        htf_bias = self._fetch_htf_bias(symbol, last_ts_m, market_tz)
        if htf_bias is None:
            return Order(symbol, None, None, None, None, "HTF bias not ready", {})

        # --- LTF Indicators ---
        atr = _atr(
            df["high"].astype(float),
            df["low"].astype(float),
            df["close"].astype(float),
            atr_period
        )

        if pd.isna(atr.iloc[-1]):
            return Order(symbol, None, None, None, None, "ATR not ready", {})

        atr_now = _as_float(atr.iloc[-1])
        
        # ATR filter
        if atr_now < min_atr:
            return Order(
                symbol, None, None, None, None,
                "ATR too low",
                {"atr": atr_now, "min_atr": min_atr}
            )

        # --- Swing detection ---
        swing_highs = _detect_swing_highs(df["high"].astype(float), swing_window)
        swing_lows = _detect_swing_lows(df["low"].astype(float), swing_window)

        # Get most recent confirmed swings
        confirmed_high_idx = swing_highs[swing_highs].index
        confirmed_low_idx = swing_lows[swing_lows].index

        if len(confirmed_high_idx) > 0:
            last_swing_high = _as_float(df.loc[confirmed_high_idx[-1], "high"])
            st.last_swing_high = last_swing_high
        
        if len(confirmed_low_idx) > 0:
            last_swing_low = _as_float(df.loc[confirmed_low_idx[-1], "low"])
            st.last_swing_low = last_swing_low

        # Current bar
        last = df.iloc[-1]
        last_close = _as_float(last["close"])
        last_open = _as_float(last["open"])

        # --- LONG BOS: Break above swing high with bullish HTF ---
        if htf_bias == "bullish" and st.last_swing_high is not None:
            # BOS trigger: close breaks above the swing high
            if last_close > st.last_swing_high and last_open <= st.last_swing_high:
                entry = last_close
                stop = st.last_swing_high - stop_atr_mult * atr_now
                
                if stop >= entry:
                    return Order(
                        symbol, None, None, None, None,
                        "invalid long stop",
                        {"stop": stop, "entry": entry}
                    )
                
                take = entry + risk_reward * (entry - stop)
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="buy",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="BOS long (swing break + bullish HTF)",
                    meta={
                        "htf_bias": htf_bias,
                        "swing_high": st.last_swing_high,
                        "atr": atr_now,
                        "trail_pct": trail_pct,
                        "trail_activate_after_partial": False,  # Trail from start
                    },
                )

        # --- SHORT BOS: Break below swing low with bearish HTF ---
        if htf_bias == "bearish" and st.last_swing_low is not None:
            # BOS trigger: close breaks below the swing low
            if last_close < st.last_swing_low and last_open >= st.last_swing_low:
                entry = last_close
                stop = st.last_swing_low + stop_atr_mult * atr_now
                
                if stop <= entry:
                    return Order(
                        symbol, None, None, None, None,
                        "invalid short stop",
                        {"stop": stop, "entry": entry}
                    )
                
                take = entry - risk_reward * (stop - entry)
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="sell",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="BOS short (swing break + bearish HTF)",
                    meta={
                        "htf_bias": htf_bias,
                        "swing_low": st.last_swing_low,
                        "atr": atr_now,
                        "trail_pct": trail_pct,
                        "trail_activate_after_partial": False,  # Trail from start
                    },
                )

        return Order(symbol, None, None, None, None, "no BOS setup", {})
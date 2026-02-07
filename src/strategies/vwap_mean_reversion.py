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
    # simplified: just track if we've seen stretch, no strict sequencing
    stretched_long: bool = False
    stretched_short: bool = False


class VWAPMeanReversion(Strategy):
    """
    VWAP Mean Reversion (intraday)
    
    Simplified logic:
      - Price stretches beyond outer band (VWAP ± band_atr_mult * ATR)
      - Then reclaims back inside the band with momentum
      - Target is VWAP
    
    No strict "must be inside first" state machine - just stretch + reclaim.
    """

    def __init__(self, name: str = "vwap_mr", **params) -> None:
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
        band_atr_mult = float(self.params.get("band_atr_mult", 1.5))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 20))
        max_minutes_after_open = int(self.params.get("max_minutes_after_open", 180))
        
        rr = float(self.params.get("rr", 1.5))
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.0))
        
        reclaim_bars = int(self.params.get("reclaim_bars", 3))
        reclaim_threshold = float(self.params.get("reclaim_threshold", 0.66))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))

        # CRITICAL FIX: Use smaller warmup for HTF
        warmup = max(atr_period + 5, 20)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume", {})

        # --- timezone conversion ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        session_open, session_close = _session_bounds(day_m, rth_open, rth_close)

        # CRITICAL FIX: Calculate indicators on TODAY'S full data (pre-market + RTH)
        # This gives enough bars for HTF, but we only TRADE during RTH
        in_today = idx_m.normalize() == day_m
        day_df = df.loc[in_today].copy()
        day_idx_m = idx_m[in_today]
        
        if day_df.empty or len(day_df) < warmup:
            return Order(symbol, None, None, None, None, "not enough bars today", {})

        # --- time window gate (ONLY TRADE DURING RTH) ---
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        mins_from_open = (last_ts_m - session_open).total_seconds() / 60.0
        if mins_from_open < min_minutes_after_open:
            return Order(symbol, None, None, None, None, "too early", {"mins_from_open": mins_from_open})
        if mins_from_open > max_minutes_after_open:
            return Order(symbol, None, None, None, None, "past MR window", {"mins_from_open": mins_from_open})

        # --- state management ---
        st = self._get_state(symbol)
        if st.last_day is None or st.last_day != day_m:
            st.last_day = day_m
            st.trades_today = 0
            st.stretched_long = False
            st.stretched_short = False

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {"trades_today": st.trades_today})

        # --- indicators ---
        vwap = _vwap_session(day_df)
        atr = _atr(day_df, atr_period)

        if pd.isna(vwap.iloc[-1]) or pd.isna(atr.iloc[-1]):
            return Order(symbol, None, None, None, None, "vwap/atr not ready", {})

        v = _as_float(vwap.iloc[-1])
        a = _as_float(atr.iloc[-1])

        if a <= 0:
            return Order(symbol, None, None, None, None, "invalid ATR", {"atr": a})

        upper_band = v + band_atr_mult * a
        lower_band = v - band_atr_mult * a

        # --- recent bars for reclaim ---
        if len(day_df) < reclaim_bars + 1:
            return Order(symbol, None, None, None, None, "not enough bars for reclaim", {})

        recent = day_df.iloc[-reclaim_bars:]
        lows = recent["low"].astype(float).values
        highs = recent["high"].astype(float).values
        closes = recent["close"].astype(float).values
        opens = recent["open"].astype(float).values

        last_close = _as_float(day_df["close"].iloc[-1])
        last_low = _as_float(day_df["low"].iloc[-1])
        last_high = _as_float(day_df["high"].iloc[-1])

        # --- detect stretch ---
        prev_close = _as_float(day_df["close"].iloc[-2])
        
        if prev_close < lower_band:
            st.stretched_long = True
        if prev_close > upper_band:
            st.stretched_short = True

        # --- LONG setup: stretched below, now reclaim with momentum ---
        if st.stretched_long:
            # reclaim: touched below band, now close back above
            touched_below = min(lows) <= lower_band
            reclaimed_above = last_close > lower_band
            
            # momentum: majority of recent bars are bullish
            bullish_count = sum(closes[i] > opens[i] for i in range(len(closes)))
            has_momentum = (bullish_count / len(closes)) >= reclaim_threshold
            
            if touched_below and reclaimed_above and has_momentum:
                entry = last_close
                stop = min(last_low, lower_band) - stop_atr_mult * a
                
                if stop >= entry:
                    return Order(symbol, None, None, None, None, "invalid long stop", {"stop": stop, "entry": entry})
                
                # target VWAP (mean reversion), fallback to RR
                take = v
                if take <= entry:
                    take = entry + rr * (entry - stop)
                
                st.stretched_long = False
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="buy",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="VWAP MR long (stretch->reclaim)",
                    meta={
                        "vwap": v,
                        "atr": a,
                        "lower_band": lower_band,
                        "upper_band": upper_band,
                        "bullish_ratio": bullish_count / len(closes),
                        "mins_from_open": mins_from_open,
                    },
                )

        # --- SHORT setup: stretched above, now reclaim with momentum ---
        if st.stretched_short:
            touched_above = max(highs) >= upper_band
            reclaimed_below = last_close < upper_band
            
            bearish_count = sum(closes[i] < opens[i] for i in range(len(closes)))
            has_momentum = (bearish_count / len(closes)) >= reclaim_threshold
            
            if touched_above and reclaimed_below and has_momentum:
                entry = last_close
                stop = max(last_high, upper_band) + stop_atr_mult * a
                
                if stop <= entry:
                    return Order(symbol, None, None, None, None, "invalid short stop", {"stop": stop, "entry": entry})
                
                take = v
                if take >= entry:
                    take = entry - rr * (stop - entry)
                
                st.stretched_short = False
                st.trades_today += 1
                
                return Order(
                    symbol=symbol,
                    side="sell",
                    entry=entry,
                    stop=stop,
                    take=take,
                    reason="VWAP MR short (stretch->reclaim)",
                    meta={
                        "vwap": v,
                        "atr": a,
                        "lower_band": lower_band,
                        "upper_band": upper_band,
                        "bearish_ratio": bearish_count / len(closes),
                        "mins_from_open": mins_from_open,
                    },
                )

        return Order(symbol, None, None, None, None, "no MR setup", {})
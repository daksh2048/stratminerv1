from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _vwap_intraday(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].astype(float)
    v = df["volume"].astype(float).replace(0, pd.NA)
    vwap = pv.cumsum() / v.cumsum()
    return vwap.ffill().infer_objects(copy=False)


@dataclass
class TPState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0


class VWAPTrendPullback(Strategy):
    """
    VWAP Trend Pullback - Simplified & HTF-friendly
    
    Logic:
      1. Price above VWAP + EMA (uptrend) or below (downtrend)
      2. Price pulls back to touch/cross VWAP
      3. Price reclaims with directional momentum
      4. Optional: trend strength filter (EMA slope)
    
    Changes from v4:
      - Removed HTF filter (confusing on HTF itself)
      - Looser pullback requirements (just needs to touch VWAP)
      - Trend filter optional and scaled by timeframe
      - Simpler continuation logic
    """

    def __init__(self, name: str = "vwap_tp", **params) -> None:
        super().__init__(name, **params)
        self._state: dict[str, TPState] = {}

    def _get_state(self, symbol: str) -> TPState:
        if symbol not in self._state:
            self._state[symbol] = TPState()
        return self._state[symbol]

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- params ---
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))
        
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 15))
        entry_cutoff_minutes = int(self.params.get("entry_cutoff_minutes", 240))
        
        ema_period = int(self.params.get("ema_period", 20))
        atr_period = int(self.params.get("atr_period", 14))
        
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.2))
        risk_reward = float(self.params.get("risk_reward", 2.0))
        
        # trend filter (optional, disabled by default for HTF)
        use_trend_filter = bool(self.params.get("use_trend_filter", False))
        ema_slope_min_atr = float(self.params.get("ema_slope_min_atr", 0.08))
        
        # pullback & reclaim
        pullback_bars = int(self.params.get("pullback_bars", 4))
        reclaim_threshold = float(self.params.get("reclaim_threshold", 0.66))
        
        max_trades_per_day = int(self.params.get("max_trades_per_day", 2))

        # CRITICAL FIX: Smaller warmup for HTF
        warmup = max(ema_period, atr_period) + 5
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        if "volume" not in df.columns:
            return Order(symbol, None, None, None, None, "missing volume", {})

        # --- timezone ---
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()
        
        session_open, session_close = _session_bounds(day_m, rth_open, rth_close)

        # CRITICAL FIX: Calculate indicators on TODAY'S full data
        in_today = idx_m.normalize() == day_m
        day_df = df.loc[in_today].copy()
        
        if day_df.empty or len(day_df) < warmup:
            return Order(symbol, None, None, None, None, "not enough bars today", {})

        # --- time gates (ONLY TRADE DURING RTH) ---
        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})
        
        if last_ts_m < (session_open + pd.Timedelta(minutes=min_minutes_after_open)):
            return Order(symbol, None, None, None, None, "too early", {})

        if last_ts_m > (session_open + pd.Timedelta(minutes=entry_cutoff_minutes)):
            return Order(symbol, None, None, None, None, "past cutoff", {})

        # --- state ---
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {"trades_today": st.trades_today})

        # --- indicators ---
        vwap = _vwap_intraday(day_df)
        ema = _ema(day_df["close"].astype(float), ema_period)
        atr = _atr(
            day_df["high"].astype(float),
            day_df["low"].astype(float),
            day_df["close"].astype(float),
            atr_period
        )

        if pd.isna(vwap.iloc[-1]) or pd.isna(ema.iloc[-1]) or pd.isna(atr.iloc[-1]):
            return Order(symbol, None, None, None, None, "indicators not ready", {})

        vwap_now = _as_float(vwap.iloc[-1])
        ema_now = _as_float(ema.iloc[-1])
        ema_prev = _as_float(ema.iloc[-2])
        atr_now = _as_float(atr.iloc[-1])

        if atr_now <= 0:
            return Order(symbol, None, None, None, None, "invalid ATR", {"atr": atr_now})

        last = day_df.iloc[-1]
        last_close = _as_float(last["close"])
        last_open = _as_float(last["open"])
        last_low = _as_float(last["low"])
        last_high = _as_float(last["high"])

        # --- trend bias ---
        uptrend = (last_close > vwap_now) and (last_close > ema_now)
        downtrend = (last_close < vwap_now) and (last_close < ema_now)

        # --- optional trend strength filter ---
        if use_trend_filter:
            ema_slope_atr = (ema_now - ema_prev) / atr_now
            
            if uptrend and ema_slope_atr < ema_slope_min_atr:
                return Order(symbol, None, None, None, None, "trend too weak (long)", {"slope": ema_slope_atr})
            if downtrend and ema_slope_atr > -ema_slope_min_atr:
                return Order(symbol, None, None, None, None, "trend too weak (short)", {"slope": ema_slope_atr})

        # --- pullback: recent bars touched VWAP ---
        lb = max(2, pullback_bars)
        recent = day_df.iloc[-lb:]
        recent_low = _as_float(recent["low"].min())
        recent_high = _as_float(recent["high"].max())

        pulled_to_vwap_long = recent_low <= vwap_now
        pulled_to_vwap_short = recent_high >= vwap_now

        # --- reclaim: current bar crosses back with momentum ---
        long_reclaim = (last_low <= vwap_now) and (last_close > vwap_now) and (last_close > last_open)
        short_reclaim = (last_high >= vwap_now) and (last_close < vwap_now) and (last_close < last_open)

        # --- continuation: majority of recent bars directional ---
        closes = recent["close"].astype(float).values
        opens = recent["open"].astype(float).values
        
        bullish_count = sum(closes[i] > opens[i] for i in range(len(closes)))
        bearish_count = sum(closes[i] < opens[i] for i in range(len(closes)))
        
        long_momentum = (bullish_count / len(closes)) >= reclaim_threshold
        short_momentum = (bearish_count / len(closes)) >= reclaim_threshold

        # --- LONG setup ---
        if uptrend and pulled_to_vwap_long and long_reclaim and long_momentum:
            entry = last_close
            stop = min(last_low, vwap_now) - stop_atr_mult * atr_now
            
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop", {"stop": stop, "entry": entry})
            
            take = entry + risk_reward * (entry - stop)
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="VWAP TP long (trend+pullback+reclaim)",
                meta={
                    "vwap": vwap_now,
                    "ema": ema_now,
                    "atr": atr_now,
                    "bullish_ratio": bullish_count / len(closes),
                },
            )

        # --- SHORT setup ---
        if downtrend and pulled_to_vwap_short and short_reclaim and short_momentum:
            entry = last_close
            stop = max(last_high, vwap_now) + stop_atr_mult * atr_now
            
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop", {"stop": stop, "entry": entry})
            
            take = entry - risk_reward * (stop - entry)
            st.trades_today += 1
            
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="VWAP TP short (trend+pullback+reclaim)",
                meta={
                    "vwap": vwap_now,
                    "ema": ema_now,
                    "atr": atr_now,
                    "bearish_ratio": bearish_count / len(closes),
                },
            )

        return Order(symbol, None, None, None, None, "no TP setup", {})
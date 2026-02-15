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
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


@dataclass
class ORBState:
    day: Optional[pd.Timestamp] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_formed: bool = False
    traded_today: bool = False
    
    # Retest state machine
    pending_long: bool = False
    pending_long_level: Optional[float] = None
    pending_long_stop: Optional[float] = None
    pending_long_bars_waiting: int = 0
    
    pending_short: bool = False
    pending_short_level: Optional[float] = None
    pending_short_stop: Optional[float] = None
    pending_short_bars_waiting: int = 0
    
    # HTF data
    htf_df: Optional[pd.DataFrame] = None
    htf_fetched: bool = False


class OpeningRangeBreakout(Strategy):
    """
    Opening Range Breakout v2 - Actually Tradeable
    
    FIXES:
    1. HTF bias filter - only trade with the trend
    2. ATR-based stops - not OR range (prevents massive stops on volatile days)
    3. Volume filter - breakout must have >1.5x average volume
    4. Retest entry - wait for pullback, don't chase
    5. Partial TP + runner - lock in 1.5R, trail to 3R
    6. Time-based OR - first 30min, not first 6 bars
    7. All trailing modes wired
    
    Parameters:
        opening_range_minutes: 30 (first 30min of RTH)
        min_minutes_after_or: 0 (allow immediate breakout)
        max_minutes_after_or: 240 (stop entries 4h after open)
        
        htf_timeframe: "1h"
        htf_ema_period: 50
        htf_period: "5y"
        
        atr_period: 14
        stop_atr_mult: 1.5 (stop = entry ± 1.5×ATR)
        min_atr: 0.10
        
        volume_filter: true
        volume_ema_period: 20
        volume_mult: 1.5 (breakout vol > 1.5x avg)
        
        use_retest: true
        retest_max_bars: 20
        retest_tolerance_atr: 0.3
        
        partial_take_pct: 0.5
        partial_take_rr: 1.5
        risk_reward: 3.0 (runner target)
        
        trail_mode: "tiered" (or atr, percent, chandelier, structure, hybrid, parabolic)
    """

    def __init__(self, name: str = "orb", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, ORBState] = {}

    def _get_state(self, symbol: str) -> ORBState:
        if symbol not in self._state:
            self._state[symbol] = ORBState()
        return self._state[symbol]

    def _get_htf_bias(
        self,
        st: ORBState,
        symbol: str,
        ltf_last_ts: pd.Timestamp,
        exchange: str = "yahoo",
    ) -> Optional[str]:
        """Get HTF trend bias - reused from swing_bos"""
        htf_timeframe = str(self.params.get("htf_timeframe", "1h"))
        htf_ema_period = int(self.params.get("htf_ema_period", 50))
        htf_period = str(self.params.get("htf_period", "60d"))

        if not st.htf_fetched:
            try:
                print(f"  [ORB] Fetching HTF ({htf_timeframe}) for {symbol}...", end="", flush=True)
                feed = CandleFeed(exchange=exchange, symbol=symbol, timeframe=htf_timeframe)
                htf_df = feed.fetch(period=htf_period)
                st.htf_df = htf_df
                st.htf_fetched = True
                print(f" done ({len(htf_df)} bars)")
            except Exception as e:
                print(f" FAILED: {e}")
                st.htf_fetched = True
                return None

        if st.htf_df is None or st.htf_df.empty:
            return None

        # Slice to current LTF timestamp (no look-ahead)
        htf_idx = st.htf_df.index
        if getattr(htf_idx, "tz", None) is None:
            htf_idx_utc = htf_idx.tz_localize("UTC")
        else:
            htf_idx_utc = htf_idx.tz_convert("UTC")

        if ltf_last_ts.tzinfo is None:
            ltf_utc = ltf_last_ts.tz_localize("UTC")
        else:
            ltf_utc = ltf_last_ts.tz_convert("UTC")

        mask = htf_idx_utc <= ltf_utc
        htf_slice = st.htf_df.loc[mask]

        if htf_slice.empty or len(htf_slice) < htf_ema_period + 4:
            return None

        htf_ema = _ema(htf_slice["close"].astype(float), htf_ema_period)

        if pd.isna(htf_ema.iloc[-1]) or len(htf_ema) < 5:
            return None

        last_close = _as_float(htf_slice["close"].iloc[-1])
        last_ema = _as_float(htf_ema.iloc[-1])
        prev_ema = _as_float(htf_ema.iloc[-4])

        ema_rising = last_ema > prev_ema
        ema_falling = last_ema < prev_ema

        if last_close > last_ema and ema_rising:
            return "bullish"
        elif last_close < last_ema and ema_falling:
            return "bearish"

        return None

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # ── Parameters ────────────────────────────────────────────────
        market_tz = str(self.params.get("market_tz", "America/New_York"))
        rth_open = str(self.params.get("rth_open", "09:30"))
        rth_close = str(self.params.get("rth_close", "16:00"))

        opening_range_minutes = int(self.params.get("opening_range_minutes", 30))
        min_minutes_after_or = int(self.params.get("min_minutes_after_or", 0))
        max_minutes_after_or = int(self.params.get("max_minutes_after_or", 240))

        exchange_for_htf = str(self.params.get("exchange", "yahoo"))
        
        atr_period = int(self.params.get("atr_period", 14))
        stop_atr_mult = float(self.params.get("stop_atr_mult", 1.5))
        min_atr = float(self.params.get("min_atr", 0.10))

        volume_filter = bool(self.params.get("volume_filter", True))
        volume_ema_period = int(self.params.get("volume_ema_period", 20))
        volume_mult = float(self.params.get("volume_mult", 1.5))

        use_retest = bool(self.params.get("use_retest", True))
        retest_max_bars = int(self.params.get("retest_max_bars", 20))
        retest_tolerance_atr = float(self.params.get("retest_tolerance_atr", 0.3))

        partial_take_pct = float(self.params.get("partial_take_pct", 0.5))
        partial_take_rr = float(self.params.get("partial_take_rr", 1.5))
        risk_reward = float(self.params.get("risk_reward", 3.0))
        trail_mode = str(self.params.get("trail_mode", "tiered"))
        trail_activate_after_partial = bool(self.params.get("trail_activate_after_partial", True))

        # ── Warmup ────────────────────────────────────────────────────
        warmup = max(atr_period + 20, volume_ema_period + 20, 60)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        # ── Timezone ──────────────────────────────────────────────────
        idx_m = _to_market_tz(df.index, market_tz)
        last_ts_m = idx_m[-1]
        day_m = last_ts_m.normalize()

        oh, om = map(int, rth_open.split(":"))
        ch, cm = map(int, rth_close.split(":"))
        session_open = day_m + pd.Timedelta(hours=oh, minutes=om)
        session_close = day_m + pd.Timedelta(hours=ch, minutes=cm)

        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})

        # ── Day reset ─────────────────────────────────────────────────
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.or_high = None
            st.or_low = None
            st.or_formed = False
            st.traded_today = False
            st.pending_long = False
            st.pending_long_level = None
            st.pending_long_stop = None
            st.pending_long_bars_waiting = 0
            st.pending_short = False
            st.pending_short_level = None
            st.pending_short_stop = None
            st.pending_short_bars_waiting = 0

        if st.traded_today:
            return Order(symbol, None, None, None, None, "already traded today", {})

        # ── HTF Bias ──────────────────────────────────────────────────
        htf_bias = self._get_htf_bias(st, symbol, last_ts_m, exchange_for_htf)

        # Cancel pending if bias flips
        if htf_bias != "bullish" and st.pending_long:
            st.pending_long = False
            st.pending_long_level = None
            st.pending_long_stop = None
            st.pending_long_bars_waiting = 0

        if htf_bias != "bearish" and st.pending_short:
            st.pending_short = False
            st.pending_short_level = None
            st.pending_short_stop = None
            st.pending_short_bars_waiting = 0

        # ── ATR ───────────────────────────────────────────────────────
        atr = _atr(
            df["high"].astype(float),
            df["low"].astype(float),
            df["close"].astype(float),
            atr_period,
        )
        if pd.isna(atr.iloc[-1]):
            return Order(symbol, None, None, None, None, "ATR not ready", {})

        atr_now = _as_float(atr.iloc[-1])
        if atr_now < min_atr:
            return Order(symbol, None, None, None, None, "ATR too low", {"atr": atr_now})

        # ── Volume EMA ────────────────────────────────────────────────
        vol_ema_val: Optional[float] = None
        if volume_filter and "volume" in df.columns:
            v_ema = _ema(df["volume"].astype(float), volume_ema_period)
            if not pd.isna(v_ema.iloc[-1]):
                vol_ema_val = _as_float(v_ema.iloc[-1])

        # ── Current bar ───────────────────────────────────────────────
        last = df.iloc[-1]
        last_close = _as_float(last["close"])
        last_open = _as_float(last["open"])
        last_high = _as_float(last["high"])
        last_low = _as_float(last["low"])
        last_vol = _as_float(last["volume"]) if "volume" in last.index else None

        # ── Opening Range Formation ───────────────────────────────────
        or_end = session_open + pd.Timedelta(minutes=opening_range_minutes)
        
        if not st.or_formed:
            # Wait until OR period is complete
            if last_ts_m < or_end:
                return Order(symbol, None, None, None, None, "OR forming", {})
            
            # Calculate OR from today's data
            in_today = (idx_m.normalize() == day_m)
            in_or = (idx_m >= session_open) & (idx_m < or_end)
            or_bars = df.loc[in_today & in_or]
            
            if len(or_bars) < 3:
                return Order(symbol, None, None, None, None, "insufficient OR bars", {})
            
            st.or_high = _as_float(or_bars["high"].max())
            st.or_low = _as_float(or_bars["low"].min())
            st.or_formed = True

        # ── Entry window ──────────────────────────────────────────────
        mins_from_or_end = (last_ts_m - or_end).total_seconds() / 60.0
        
        if mins_from_or_end < min_minutes_after_or:
            return Order(symbol, None, None, None, None, "too soon after OR", {})
        
        if mins_from_or_end > max_minutes_after_or:
            return Order(symbol, None, None, None, None, "past entry window", {})

        # ── Helper to build order meta ────────────────────────────────
        def _order_meta(bias, stop):
            meta = {
                "htf_bias": bias,
                "or_high": st.or_high,
                "or_low": st.or_low,
                "atr": atr_now,
                "trail_mode": trail_mode,
                "trail_activate_after_partial": trail_activate_after_partial,
                "partial_take_pct": partial_take_pct,
                "partial_take_rr": partial_take_rr,
                "move_stop_to_be": True,
                "runner_take_mode": "rr",
                "runner_rr": risk_reward,
                "initial_stop": stop,
            }
            
            # Wire all trailing mode params
            if "trail_pct" in self.params:
                meta["trail_pct"] = float(self.params["trail_pct"])
            if "trail_atr_mult" in self.params:
                meta["trail_atr_mult"] = float(self.params["trail_atr_mult"])
            if "trail_lookback" in self.params:
                meta["trail_lookback"] = int(self.params["trail_lookback"])
            if "trail_tier1_pct" in self.params:
                meta["trail_tier1_pct"] = float(self.params["trail_tier1_pct"])
            if "trail_tier2_pct" in self.params:
                meta["trail_tier2_pct"] = float(self.params["trail_tier2_pct"])
            if "trail_tier3_pct" in self.params:
                meta["trail_tier3_pct"] = float(self.params["trail_tier3_pct"])
            if "trail_swing_window" in self.params:
                meta["trail_swing_window"] = int(self.params["trail_swing_window"])
            if "trail_buffer_mult" in self.params:
                meta["trail_buffer_mult"] = float(self.params["trail_buffer_mult"])
            if "trail_initial_pct" in self.params:
                meta["trail_initial_pct"] = float(self.params["trail_initial_pct"])
            if "trail_acceleration" in self.params:
                meta["trail_acceleration"] = float(self.params["trail_acceleration"])
            if "trail_min_pct" in self.params:
                meta["trail_min_pct"] = float(self.params["trail_min_pct"])
            
            return meta

        # ══════════════════════════════════════════════════════════════
        # RETEST STATE MACHINE
        # ══════════════════════════════════════════════════════════════

        # ── Pending LONG retest ───────────────────────────────────────
        if st.pending_long and st.pending_long_level is not None:
            level = st.pending_long_level
            tolerance = retest_tolerance_atr * atr_now
            st.pending_long_bars_waiting += 1

            # Invalidate: price went back through OR
            if last_close < level - tolerance:
                st.pending_long = False
                st.pending_long_level = None
                st.pending_long_stop = None
                st.pending_long_bars_waiting = 0

            # Timeout
            elif st.pending_long_bars_waiting > retest_max_bars:
                st.pending_long = False
                st.pending_long_level = None
                st.pending_long_stop = None
                st.pending_long_bars_waiting = 0

            # Retest: touched back and bullish rejection
            elif last_low <= level + tolerance and last_close > last_open:
                entry = last_close
                stop = st.pending_long_stop

                if stop is not None and stop < entry:
                    partial_tp = entry + partial_take_rr * (entry - stop)
                    st.traded_today = True
                    st.pending_long = False
                    st.pending_long_level = None
                    st.pending_long_stop = None
                    st.pending_long_bars_waiting = 0

                    return Order(
                        symbol=symbol,
                        side="buy",
                        entry=entry,
                        stop=stop,
                        take=partial_tp,
                        reason="ORB long — retest entry",
                        meta=_order_meta(htf_bias, stop),
                    )

        # ── Pending SHORT retest ──────────────────────────────────────
        if st.pending_short and st.pending_short_level is not None:
            level = st.pending_short_level
            tolerance = retest_tolerance_atr * atr_now
            st.pending_short_bars_waiting += 1

            if last_close > level + tolerance:
                st.pending_short = False
                st.pending_short_level = None
                st.pending_short_stop = None
                st.pending_short_bars_waiting = 0

            elif st.pending_short_bars_waiting > retest_max_bars:
                st.pending_short = False
                st.pending_short_level = None
                st.pending_short_stop = None
                st.pending_short_bars_waiting = 0

            elif last_high >= level - tolerance and last_close < last_open:
                entry = last_close
                stop = st.pending_short_stop

                if stop is not None and stop > entry:
                    partial_tp = entry - partial_take_rr * (stop - entry)
                    st.traded_today = True
                    st.pending_short = False
                    st.pending_short_level = None
                    st.pending_short_stop = None
                    st.pending_short_bars_waiting = 0

                    return Order(
                        symbol=symbol,
                        side="sell",
                        entry=entry,
                        stop=stop,
                        take=partial_tp,
                        reason="ORB short — retest entry",
                        meta=_order_meta(htf_bias, stop),
                    )

        # ══════════════════════════════════════════════════════════════
        # BREAKOUT DETECTION
        # ══════════════════════════════════════════════════════════════

        if htf_bias is None:
            return Order(symbol, None, None, None, None, "HTF no clear bias", {})

        # ── LONG breakout ─────────────────────────────────────────────
        if htf_bias == "bullish" and last_close > st.or_high and last_open <= st.or_high:
            # Volume filter
            if volume_filter and vol_ema_val is not None and last_vol is not None:
                if last_vol < vol_ema_val * volume_mult:
                    return Order(symbol, None, None, None, None, "low volume breakout", {})

            # ATR-based stop (not OR low)
            stop = last_close - stop_atr_mult * atr_now
            
            if stop >= last_close:
                return Order(symbol, None, None, None, None, "invalid long stop", {})

            if use_retest:
                if not st.pending_long:
                    st.pending_long = True
                    st.pending_long_level = st.or_high
                    st.pending_long_stop = stop
                    st.pending_long_bars_waiting = 0
                return Order(symbol, None, None, None, None, "ORB long — waiting for retest", {})
            else:
                entry = last_close
                partial_tp = entry + partial_take_rr * (entry - stop)
                st.traded_today = True
                return Order(
                    symbol=symbol,
                    side="buy",
                    entry=entry,
                    stop=stop,
                    take=partial_tp,
                    reason="ORB long — immediate entry",
                    meta=_order_meta(htf_bias, stop),
                )

        # ── SHORT breakout ────────────────────────────────────────────
        if htf_bias == "bearish" and last_close < st.or_low and last_open >= st.or_low:
            if volume_filter and vol_ema_val is not None and last_vol is not None:
                if last_vol < vol_ema_val * volume_mult:
                    return Order(symbol, None, None, None, None, "low volume breakout", {})

            stop = last_close + stop_atr_mult * atr_now
            
            if stop <= last_close:
                return Order(symbol, None, None, None, None, "invalid short stop", {})

            if use_retest:
                if not st.pending_short:
                    st.pending_short = True
                    st.pending_short_level = st.or_low
                    st.pending_short_stop = stop
                    st.pending_short_bars_waiting = 0
                return Order(symbol, None, None, None, None, "ORB short — waiting for retest", {})
            else:
                entry = last_close
                partial_tp = entry - partial_take_rr * (stop - entry)
                st.traded_today = True
                return Order(
                    symbol=symbol,
                    side="sell",
                    entry=entry,
                    stop=stop,
                    take=partial_tp,
                    reason="ORB short — immediate entry",
                    meta=_order_meta(htf_bias, stop),
                )

        return Order(symbol, None, None, None, None, "no ORB setup", {})
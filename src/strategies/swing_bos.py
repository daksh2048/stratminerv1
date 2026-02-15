from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
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


def _detect_swing_highs(high: pd.Series, window: int) -> pd.Series:
    """
    Vectorized swing high detection.
    Call this on df.iloc[:-window] to eliminate look-ahead bias.
    The shift(-window) is then bounded by the truncated series.
    """
    roll_left = high.shift(1).rolling(window).max()
    roll_right = high.shift(-window).rolling(window).max()
    return (high > roll_left) & (high > roll_right)


def _detect_swing_lows(low: pd.Series, window: int) -> pd.Series:
    roll_left = low.shift(1).rolling(window).min()
    roll_right = low.shift(-window).rolling(window).min()
    return (low < roll_left) & (low < roll_right)


@dataclass
class BOSState:
    day: Optional[pd.Timestamp] = None
    trades_today: int = 0

    # Most recent confirmed swing levels
    last_swing_high: Optional[float] = None
    last_swing_high_ts: Optional[object] = None
    last_swing_low: Optional[float] = None
    last_swing_low_ts: Optional[object] = None

    # The swing LOW that preceded the swing high (used as stop for long BOS)
    swing_low_before_high: Optional[float] = None
    # The swing HIGH that preceded the swing low (used as stop for short BOS)
    swing_high_before_low: Optional[float] = None

    # Retest pending state machine — LONG
    pending_long: bool = False
    pending_long_bos_level: Optional[float] = None
    pending_long_stop: Optional[float] = None
    pending_long_bars_waiting: int = 0

    # Retest pending state machine — SHORT
    pending_short: bool = False
    pending_short_bos_level: Optional[float] = None
    pending_short_stop: Optional[float] = None
    pending_short_bars_waiting: int = 0

    # HTF: fetched ONCE per symbol per backtest run
    htf_df: Optional[pd.DataFrame] = None
    htf_fetched: bool = False


class SwingBOS(Strategy):
    """
    Swing Break of Structure (BOS) v2

    What changed vs v1:
    ─────────────────────────────────────────────────────────────────────────
    FIX 1 – HTF look-ahead bias eliminated
        HTF data is fetched ONCE per symbol. On every bar we slice the HTF
        DataFrame to rows whose timestamp <= current LTF bar before calculating
        the EMA. The old code used iloc[-1] on the full 60-day dataset, which
        meant every historical bar was reading the future.

    FIX 2 – Swing detection look-ahead bias eliminated
        The old vectorized detector used shift(-window) on the full df, which
        peeks `window` bars into the future to confirm a swing high/low. We now
        call the detector on df.iloc[:-swing_window], so the right-side window
        is always bounded by past data only.

    FIX 3 – Stop placement corrected
        Long BOS stop was placed at (swing_high - ATR_mult), which puts the
        stop inside the range that was just broken. Correct placement is below
        the swing LOW that PRECEDED the broken swing high — that is where
        structure is actually invalidated. We track this "preceding swing" and
        fall back to the broken level minus 1.5×ATR only if no preceding swing
        is available.

    FIX 4 – Trail mode wired correctly
        The old code set trail_pct=0.02 but never set trail_mode, so the broker
        silently fell through to percent trailing. We now set trail_mode="atr"
        and trail_atr_mult, using the broker's advanced trailing path.

    UPGRADE 1 – Retest entry (optional, default ON)
        After detecting a BOS, the strategy sets a "pending" state and waits for
        price to pull back to the broken level and show a rejection candle before
        entering. This tightens R and avoids chasing. Configurable via
        use_retest, retest_max_bars, retest_tolerance_atr.

    UPGRADE 2 – Volume confirmation (optional, default ON)
        The breakout bar must have volume > volume_mult × EMA(volume). Low-volume
        breakouts are skipped entirely.

    UPGRADE 3 – Partial take + runner
        50% of position closed at partial_take_rr (default 1.5R). Stop moved to
        breakeven. Remaining 50% trailed with ATR-based stop until full_take_rr
        (default 2.5R) or stop out.

    UPGRADE 4 – Structure invalidation for pending trades
        If price recrosses back through the BOS level (beyond tolerance) while
        we're waiting for a retest, the pending state is cancelled immediately.

    Parameters
    ──────────
    htf_timeframe          : "1h"
    htf_ema_period         : 20
    swing_window           : 5     (bars on each side for swing confirmation)
    atr_period             : 14
    stop_atr_mult          : 0.3   (buffer below/above the preceding swing)
    risk_reward            : 2.5   (runner take level)
    trail_atr_mult         : 2.0
    min_atr                : 0.0
    max_trades_per_day     : 2
    min_minutes_after_open : 30
    entry_cutoff_minutes   : 240
    use_retest             : True
    retest_max_bars        : 10
    retest_tolerance_atr   : 0.5
    use_volume_filter      : True
    volume_ema_period      : 20
    volume_mult            : 1.2
    partial_take_pct       : 0.5
    partial_take_rr        : 1.5
    """

    def __init__(self, name: str = "swing_bos", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, BOSState] = {}

    def _get_state(self, symbol: str) -> BOSState:
        if symbol not in self._state:
            self._state[symbol] = BOSState()
        return self._state[symbol]

    # ------------------------------------------------------------------
    # HTF bias — fetched once, sliced at runtime (no look-ahead)
    # ------------------------------------------------------------------

    def _get_htf_bias(
        self,
        st: BOSState,
        symbol: str,
        ltf_last_ts: pd.Timestamp,
        exchange: str = "yahoo",
    ) -> Optional[str]:
        htf_timeframe = str(self.params.get("htf_timeframe", "1h"))
        htf_ema_period = int(self.params.get("htf_ema_period", 20))
        htf_period = str(self.params.get("htf_period", "60d"))

        # Fetch ONCE per backtest run per symbol
        if not st.htf_fetched:
            try:
                print(f"  [BOS] Fetching HTF ({htf_timeframe}) for {symbol}...", end="", flush=True)
                feed = CandleFeed(exchange=exchange, symbol=symbol, timeframe=htf_timeframe)
                htf_df = feed.fetch(period=htf_period)
                st.htf_df = htf_df
                st.htf_fetched = True
                print(f" done ({len(htf_df)} bars)")
            except Exception as e:
                print(f" FAILED: {e}")
                st.htf_fetched = True  # don't retry every bar on failure
                return None

        if st.htf_df is None or st.htf_df.empty:
            return None

        # ── CRITICAL: slice to current LTF timestamp ──────────────────────
        # Normalise both index and ltf_last_ts to UTC before comparing
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
        last_ema   = _as_float(htf_ema.iloc[-1])
        prev_ema   = _as_float(htf_ema.iloc[-4])

        ema_rising  = last_ema > prev_ema
        ema_falling = last_ema < prev_ema

        if last_close > last_ema and ema_rising:
            return "bullish"
        elif last_close < last_ema and ema_falling:
            return "bearish"

        return None

    # ------------------------------------------------------------------
    # Main strategy loop
    # ------------------------------------------------------------------

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # ── Parameters ────────────────────────────────────────────────
        market_tz              = str(self.params.get("market_tz", "America/New_York"))
        rth_open               = str(self.params.get("rth_open", "09:30"))
        rth_close              = str(self.params.get("rth_close", "16:00"))

        swing_window           = int(self.params.get("swing_window", 5))
        atr_period             = int(self.params.get("atr_period", 14))

        stop_atr_mult          = float(self.params.get("stop_atr_mult", 0.3))
        risk_reward            = float(self.params.get("risk_reward", 2.5))
        trail_mode             = str(self.params.get("trail_mode", "atr"))
        trail_atr_mult         = float(self.params.get("trail_atr_mult", 2.0))

        min_atr                = float(self.params.get("min_atr", 0.0))
        max_trades_per_day     = int(self.params.get("max_trades_per_day", 2))
        min_minutes_after_open = int(self.params.get("min_minutes_after_open", 30))
        entry_cutoff_minutes   = int(self.params.get("entry_cutoff_minutes", 240))

        use_retest             = bool(self.params.get("use_retest", True))
        retest_max_bars        = int(self.params.get("retest_max_bars", 10))
        retest_tolerance_atr   = float(self.params.get("retest_tolerance_atr", 0.5))

        use_volume_filter      = bool(self.params.get("use_volume_filter", True))
        volume_ema_period      = int(self.params.get("volume_ema_period", 20))
        volume_mult            = float(self.params.get("volume_mult", 1.2))

        partial_take_pct       = float(self.params.get("partial_take_pct", 0.5))
        partial_take_rr        = float(self.params.get("partial_take_rr", 1.5))
        trail_activate_after_partial = bool(self.params.get("trail_activate_after_partial", False))
        # max_bars_open: pass this to meta so broker can force-close stale positions.
        # Compute from entry_cutoff_minutes: bars from open until cutoff + bars from open to entry.
        # Simpler: use a fixed daily cap, e.g. 390min / 5min = 78 bars per RTH session.
        # 0 = disabled (no forced close)
        max_bars_open          = int(self.params.get("max_bars_open", 0))

        # ── Warmup ────────────────────────────────────────────────────
        warmup = max(atr_period + swing_window * 2 + 5, 50)
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        max_lookback = max(atr_period * 3, swing_window * 4 + 100)
        df = df.iloc[-max_lookback:]

        # ── Timezone ──────────────────────────────────────────────────
        idx_m      = _to_market_tz(df.index, market_tz)
        last_ts_m  = idx_m[-1]
        day_m      = last_ts_m.normalize()

        oh, om = map(int, rth_open.split(":"))
        ch, cm = map(int, rth_close.split(":"))
        session_open  = day_m + pd.Timedelta(hours=oh, minutes=om)
        session_close = day_m + pd.Timedelta(hours=ch, minutes=cm)

        if last_ts_m < session_open or last_ts_m > session_close:
            return Order(symbol, None, None, None, None, "outside RTH", {})

        mins_from_open = (last_ts_m - session_open).total_seconds() / 60.0
        if mins_from_open < min_minutes_after_open:
            return Order(symbol, None, None, None, None, "too early", {"mins_from_open": mins_from_open})
        if mins_from_open > entry_cutoff_minutes:
            return Order(symbol, None, None, None, None, "past cutoff", {"mins_from_open": mins_from_open})

        # ── Day reset ─────────────────────────────────────────────────
        st = self._get_state(symbol)
        if st.day is None or st.day != day_m:
            st.day = day_m
            st.trades_today = 0
            st.last_swing_high = None
            st.last_swing_high_ts = None
            st.last_swing_low = None
            st.last_swing_low_ts = None
            st.swing_low_before_high = None
            st.swing_high_before_low = None
            # Clear pending setups from prior day (gap open invalidates levels)
            st.pending_long = False
            st.pending_long_bos_level = None
            st.pending_long_stop = None
            st.pending_long_bars_waiting = 0
            st.pending_short = False
            st.pending_short_bos_level = None
            st.pending_short_stop = None
            st.pending_short_bars_waiting = 0

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {"trades_today": st.trades_today})

        # ── HTF Bias ──────────────────────────────────────────────────
        exchange_for_htf = str(self.params.get("exchange", "yahoo"))
        htf_bias = self._get_htf_bias(st, symbol, last_ts_m, exchange_for_htf)

        # Cancel pending trades if HTF bias flipped against them
        if htf_bias != "bullish" and st.pending_long:
            st.pending_long = False
            st.pending_long_bos_level = None
            st.pending_long_stop = None
            st.pending_long_bars_waiting = 0

        if htf_bias != "bearish" and st.pending_short:
            st.pending_short = False
            st.pending_short_bos_level = None
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
        if use_volume_filter and "volume" in df.columns:
            v_ema = _ema(df["volume"].astype(float), volume_ema_period)
            if not pd.isna(v_ema.iloc[-1]):
                vol_ema_val = _as_float(v_ema.iloc[-1])

        # ── Current bar ───────────────────────────────────────────────
        last       = df.iloc[-1]
        last_close = _as_float(last["close"])
        last_open  = _as_float(last["open"])
        last_high  = _as_float(last["high"])
        last_low   = _as_float(last["low"])
        last_vol   = _as_float(last["volume"]) if "volume" in last.index else None

        # ── Swing detection (no look-ahead) ───────────────────────────
        # Key: slice off the last swing_window bars so shift(-window) doesn't
        # read future data. Confirmed swings are at least swing_window bars old.
        if len(df) > swing_window:
            df_sw = df.iloc[:-swing_window]
        else:
            df_sw = df

        swing_highs = _detect_swing_highs(df_sw["high"].astype(float), swing_window)
        swing_lows  = _detect_swing_lows(df_sw["low"].astype(float), swing_window)

        confirmed_high_idx = swing_highs[swing_highs].index
        confirmed_low_idx  = swing_lows[swing_lows].index

        # Update swing state and track the preceding swing for stop placement
        if len(confirmed_high_idx) > 0:
            new_h_ts = confirmed_high_idx[-1]
            if st.last_swing_high_ts is None or new_h_ts != st.last_swing_high_ts:
                st.last_swing_high    = _as_float(df_sw.loc[new_h_ts, "high"])
                st.last_swing_high_ts = new_h_ts
                # Preceding swing LOW = most recent confirmed low BEFORE this high
                lows_before = [i for i in confirmed_low_idx if i < new_h_ts]
                st.swing_low_before_high = (
                    _as_float(df_sw.loc[lows_before[-1], "low"])
                    if lows_before else None
                )

        if len(confirmed_low_idx) > 0:
            new_l_ts = confirmed_low_idx[-1]
            if st.last_swing_low_ts is None or new_l_ts != st.last_swing_low_ts:
                st.last_swing_low    = _as_float(df_sw.loc[new_l_ts, "low"])
                st.last_swing_low_ts = new_l_ts
                # Preceding swing HIGH = most recent confirmed high BEFORE this low
                highs_before = [i for i in confirmed_high_idx if i < new_l_ts]
                st.swing_high_before_low = (
                    _as_float(df_sw.loc[highs_before[-1], "high"])
                    if highs_before else None
                )

        # ── Helper to build order meta ────────────────────────────────
        def _order_meta(htf_bias, bos_level, stop):
            meta = {
                "htf_bias":                 htf_bias,
                "bos_level":                bos_level,
                "atr":                      atr_now,
                # Trailing mode
                "trail_mode":               trail_mode,
                "trail_atr_mult":           trail_atr_mult,
                "trail_activate_after_partial": trail_activate_after_partial,
                # Partial TP wiring
                "partial_take_pct":         partial_take_pct,
                "partial_take_rr":          partial_take_rr,
                "move_stop_to_be":          True,
                "runner_take_mode":         "rr",
                "runner_rr":                risk_reward,
                "initial_stop":             stop,
                # Time-based expiry
                "max_bars_open":            max_bars_open if max_bars_open > 0 else None,
            }
            
            # Add all trailing mode specific params from config
            # Percent mode
            if "trail_pct" in self.params:
                meta["trail_pct"] = float(self.params["trail_pct"])
            
            # Chandelier mode
            if "trail_lookback" in self.params:
                meta["trail_lookback"] = int(self.params["trail_lookback"])
            
            # Tiered mode
            if "trail_tier1_pct" in self.params:
                meta["trail_tier1_pct"] = float(self.params["trail_tier1_pct"])
            if "trail_tier2_pct" in self.params:
                meta["trail_tier2_pct"] = float(self.params["trail_tier2_pct"])
            if "trail_tier3_pct" in self.params:
                meta["trail_tier3_pct"] = float(self.params["trail_tier3_pct"])
            
            # Structure mode
            if "trail_swing_window" in self.params:
                meta["trail_swing_window"] = int(self.params["trail_swing_window"])
            if "trail_buffer_mult" in self.params:
                meta["trail_buffer_mult"] = float(self.params["trail_buffer_mult"])
            
            # Parabolic mode
            if "trail_initial_pct" in self.params:
                meta["trail_initial_pct"] = float(self.params["trail_initial_pct"])
            if "trail_acceleration" in self.params:
                meta["trail_acceleration"] = float(self.params["trail_acceleration"])
            if "trail_min_pct" in self.params:
                meta["trail_min_pct"] = float(self.params["trail_min_pct"])
            
            return meta

        # ══════════════════════════════════════════════════════════════
        # RETEST STATE MACHINE — check pending before looking for new BOS
        # ══════════════════════════════════════════════════════════════

        # ── Pending LONG retest ───────────────────────────────────────
        if st.pending_long and st.pending_long_bos_level is not None:
            bos_level = st.pending_long_bos_level
            tolerance = retest_tolerance_atr * atr_now
            st.pending_long_bars_waiting += 1

            # Invalidate: price recrossed meaningfully back below BOS level
            if last_close < bos_level - tolerance:
                st.pending_long = False
                st.pending_long_bos_level = None
                st.pending_long_stop = None
                st.pending_long_bars_waiting = 0

            # Timeout
            elif st.pending_long_bars_waiting > retest_max_bars:
                st.pending_long = False
                st.pending_long_bos_level = None
                st.pending_long_stop = None
                st.pending_long_bars_waiting = 0

            # Retest: price touched back to BOS level and current bar is bullish
            elif last_low <= bos_level + tolerance and last_close > last_open:
                entry = last_close
                stop  = st.pending_long_stop

                if stop is not None and stop < entry:
                    partial_tp = entry + partial_take_rr * (entry - stop)
                    take       = entry + risk_reward    * (entry - stop)
                    st.trades_today += 1
                    st.pending_long = False
                    st.pending_long_bos_level = None
                    st.pending_long_stop = None
                    st.pending_long_bars_waiting = 0

                    return Order(
                        symbol=symbol,
                        side="buy",
                        entry=entry,
                        stop=stop,
                        take=partial_tp,   # first take = partial (1.5R)
                        reason="BOS long — retest entry",
                        meta=_order_meta(htf_bias, bos_level, stop),
                    )

        # ── Pending SHORT retest ──────────────────────────────────────
        if st.pending_short and st.pending_short_bos_level is not None:
            bos_level = st.pending_short_bos_level
            tolerance = retest_tolerance_atr * atr_now
            st.pending_short_bars_waiting += 1

            if last_close > bos_level + tolerance:
                st.pending_short = False
                st.pending_short_bos_level = None
                st.pending_short_stop = None
                st.pending_short_bars_waiting = 0

            elif st.pending_short_bars_waiting > retest_max_bars:
                st.pending_short = False
                st.pending_short_bos_level = None
                st.pending_short_stop = None
                st.pending_short_bars_waiting = 0

            elif last_high >= bos_level - tolerance and last_close < last_open:
                entry = last_close
                stop  = st.pending_short_stop

                if stop is not None and stop > entry:
                    partial_tp = entry - partial_take_rr * (stop - entry)
                    take       = entry - risk_reward    * (stop - entry)
                    st.trades_today += 1
                    st.pending_short = False
                    st.pending_short_bos_level = None
                    st.pending_short_stop = None
                    st.pending_short_bars_waiting = 0

                    return Order(
                        symbol=symbol,
                        side="sell",
                        entry=entry,
                        stop=stop,
                        take=partial_tp,   # first take = partial (1.5R)
                        reason="BOS short — retest entry",
                        meta=_order_meta(htf_bias, bos_level, stop),
                    )

        # ══════════════════════════════════════════════════════════════
        # BOS DETECTION — only if HTF bias is clear
        # ══════════════════════════════════════════════════════════════

        if htf_bias is None:
            return Order(symbol, None, None, None, None, "HTF no clear bias", {})

        # ── LONG BOS ─────────────────────────────────────────────────
        # Two valid triggers:
        #   A) Fresh cross: prev_close was <= swing_high, current close > swing_high
        #   B) Swing just confirmed and price is ALREADY above it
        #      (swing_window bars pass before confirmation — by then the actual breakout
        #       candle is in the past and last_open check would never fire)
        prev_close = _as_float(df.iloc[-2]["close"]) if len(df) >= 2 else last_open
        long_fresh_cross    = (last_close > st.last_swing_high and prev_close <= st.last_swing_high)                               if st.last_swing_high is not None else False
        long_confirmed_late = (
            st.last_swing_high is not None
            and len(confirmed_high_idx) > 0
            and confirmed_high_idx[-1] == st.last_swing_high_ts
            and last_close > st.last_swing_high
            and not st.pending_long
        )

        if htf_bias == "bullish" and st.last_swing_high is not None:
            if (long_fresh_cross or long_confirmed_late) and not st.pending_long:

                # Volume filter on breakout bar
                if use_volume_filter and vol_ema_val is not None and last_vol is not None:
                    if last_vol < vol_ema_val * volume_mult:
                        return Order(symbol, None, None, None, None, "low volume breakout", {})

                # ── Stop: below preceding swing low, with ATR buffer ──
                if st.swing_low_before_high is not None:
                    stop = st.swing_low_before_high - stop_atr_mult * atr_now
                else:
                    # No preceding swing tracked yet — use broken level minus 1.5×ATR
                    stop = st.last_swing_high - 1.5 * atr_now

                bos_level = st.last_swing_high

                if use_retest:
                    if not st.pending_long:
                        st.pending_long = True
                        st.pending_long_bos_level = bos_level
                        st.pending_long_stop = stop
                        st.pending_long_bars_waiting = 0
                    return Order(symbol, None, None, None, None,
                                 "BOS long — waiting for retest", {"bos_level": bos_level})
                else:
                    entry = last_close
                    if stop >= entry:
                        return Order(symbol, None, None, None, None, "invalid long stop", {})
                    partial_tp = entry + partial_take_rr * (entry - stop)
                    st.trades_today += 1
                    return Order(
                        symbol=symbol,
                        side="buy",
                        entry=entry,
                        stop=stop,
                        take=partial_tp,
                        reason="BOS long — immediate entry",
                        meta=_order_meta(htf_bias, bos_level, stop),
                    )

        # ── SHORT BOS ────────────────────────────────────────────────
        short_fresh_cross    = (last_close < st.last_swing_low and prev_close >= st.last_swing_low)                                if st.last_swing_low is not None else False
        short_confirmed_late = (
            st.last_swing_low is not None
            and len(confirmed_low_idx) > 0
            and confirmed_low_idx[-1] == st.last_swing_low_ts
            and last_close < st.last_swing_low
            and not st.pending_short
        )

        if htf_bias == "bearish" and st.last_swing_low is not None:
            if (short_fresh_cross or short_confirmed_late) and not st.pending_short:

                if use_volume_filter and vol_ema_val is not None and last_vol is not None:
                    if last_vol < vol_ema_val * volume_mult:
                        return Order(symbol, None, None, None, None, "low volume breakout", {})

                # ── Stop: above preceding swing high, with ATR buffer ─
                if st.swing_high_before_low is not None:
                    stop = st.swing_high_before_low + stop_atr_mult * atr_now
                else:
                    stop = st.last_swing_low + 1.5 * atr_now

                bos_level = st.last_swing_low

                if use_retest:
                    if not st.pending_short:
                        st.pending_short = True
                        st.pending_short_bos_level = bos_level
                        st.pending_short_stop = stop
                        st.pending_short_bars_waiting = 0
                    return Order(symbol, None, None, None, None,
                                 "BOS short — waiting for retest", {"bos_level": bos_level})
                else:
                    entry = last_close
                    if stop <= entry:
                        return Order(symbol, None, None, None, None, "invalid short stop", {})
                    partial_tp = entry - partial_take_rr * (stop - entry)
                    st.trades_today += 1
                    return Order(
                        symbol=symbol,
                        side="sell",
                        entry=entry,
                        stop=stop,
                        take=partial_tp,
                        reason="BOS short — immediate entry",
                        meta=_order_meta(htf_bias, bos_level, stop),
                    )

        return Order(symbol, None, None, None, None, "no BOS setup", {})
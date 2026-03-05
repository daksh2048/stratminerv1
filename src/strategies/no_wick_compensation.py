"""
No-Wick / Compensation Play Strategy  —  Full Transcript-Compliant Implementation
====================================================================================

RULE COMPLIANCE MAP (against video transcript):

Rule 1  ✅  Only take no-wick candles WITH the trend.
             Trend = body closures only (wicks ignored completely).
             Bullish: no bottom wick in uptrend → buy.
             Bearish: no top wick in downtrend → sell.

Rule 2  ✅  Avoid big news / red-folder events.
             Cannot automate without live feed. Flag in meta. Caller must filter.
             [Known gap — requires ForexFactory feed]

Rule 3  ✅  Avoid first 3 hours of Asia open. Don't hold through NY close.
             Avoid first 15-min candle of NY open.
             All session boundaries UTC-configurable in config.yaml.

Rule 4  ✅  Max 9 candles between signal creation and tap.
             Measured by TIMESTAMP difference (not context-window index).
             Optimal 1-4 candles; 9 is hard maximum.

Rule 5  ✅  No imbalances against your position.
             Imbalance zone drawn from WICKS (high/low of imbalance candle),
             NOT from body open/close — as transcript specifies.
             Exception: if structure exists between the imbalance and current
             price, the imbalance is protected/shielded and trade is allowed.
             Exception: if imbalance is small enough that stop covers it.

Rule 6  ✅  "Almost tapped then reversed" invalidation.
             If price comes within near_miss_threshold of tap level and then
             moves to the approximate TP level without entering, signal is dead.
             Second tap on a dead signal is NOT taken.

Rule 7  ✅  Strict 1:1 R:R. Stop at wick of most recent structural pivot
             (body closures used to FIND the pivot; actual wick price used as stop).
             Breathing room applied below/above pivot wick.

Rule 8  ✅  Supported pairs: USDJPY, GBPUSD, AUDUSD, AUDJPY only.
             Breathing room per pair: USDJPY 5.5 pip, GBPUSD 4.0, AUDUSD 3.8, AUDJPY 5.0.

Rule 9  —   Higher timeframe priority: if no-wick exists on both 15min and 30min,
             use 30min. [Not implementable in single-TF call — requires caller
             to pass 30min context. Meta flag added so caller can handle this.]

Trade Mgmt ✅  If a no-wick appears against your open position, move SL/TP to
               breakeven when tapped. Implemented via `management_signals` in
               Order.meta — caller/broker must act on this.

Additional fixes vs previous version:
- Doji candles filtered from structural pivot search (large wick, tiny body).
- Imbalance zone uses full wick range (highs/lows) not body range.
- Structure-protects-imbalance logic implemented.
- Rule 6 near-miss tracking implemented.
- Timestamp-based elapsed candle count (not context-index).
- Pivot uniqueness uses left/right side comparison (not full-window np.sum==1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

from src.core.types import Order
from .base import Strategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PAIRS = {"USDJPY", "GBPUSD", "AUDUSD", "AUDJPY"}

PIP_MAP = {
    "AUDUSD": 3.8,
    "GBPUSD": 4.0,
    "USDJPY": 5.5,
    "AUDJPY": 5.0,
}

# Doji: body < this fraction of total candle range → ignored for structure
DOJI_BODY_RATIO = 0.15

# Rule 6: "almost tapped" threshold — how close price must come to tap level
# before we start watching for the TP-hit invalidation (in pip multiples)
NEAR_MISS_BREATHING_MULT = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def _to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _simple_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def _is_doji(o: float, h: float, l: float, c: float) -> bool:
    """
    Transcript: 'doji candles don't count, we disregard them.'
    A doji has a huge wick and small body relative to total range.
    """
    total_range = h - l
    if total_range <= 0:
        return True  # zero-range bar — treat as doji
    body = abs(c - o)
    return (body / total_range) < DOJI_BODY_RATIO


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NoWickCandle:
    """
    Pending no-wick signal waiting to be tapped.

    created_time: UTC timestamp of the bar that formed the no-wick candle.
                  Used to count elapsed candles — NOT an index in the context
                  window, which would always equal len(context)-1 in a sliding
                  window and cause candles_elapsed to always be 0.

    near_miss:    True once price comes within NEAR_MISS_BREATHING_MULT *
                  breathing_room of the tap level. Used for Rule 6.

    near_miss_tp: Approximate TP level computed at the moment of near-miss.
                  If price subsequently reaches this level, the signal is dead.
    """
    created_time:  pd.Timestamp
    price:         float          # tap level = open of no-wick candle
    side:          str            # 'buy' or 'sell'
    near_miss:     bool  = False
    near_miss_tp:  Optional[float] = None


@dataclass
class NoWickState:
    day:              Optional[pd.Timestamp] = None
    trades_today:     int = 0
    pending_signals:  List[NoWickCandle] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class NoWickCompensationPlay(Strategy):
    """
    No-Wick Compensation Play — implements every transcript rule.
    Drop-in replacement: copy to src/strategies/no_wick_compensation.py
    """

    def __init__(self, name: str = "no_wick", **params) -> None:
        super().__init__(name, **params)
        self._state: Dict[str, NoWickState] = {}

    def _get_state(self, symbol: str) -> NoWickState:
        if symbol not in self._state:
            self._state[symbol] = NoWickState()
        return self._state[symbol]

    # ------------------------------------------------------------------
    # Rule 1 — No-wick candle detection
    # ------------------------------------------------------------------

    def _is_no_wick_bullish(self, row) -> bool:
        """
        Bullish candle (close > open) with NO bottom wick (or < 5% of body).
        Flat bottom = open is at or very near the low.
        """
        o = _as_float(row["open"])
        l = _as_float(row["low"])
        c = _as_float(row["close"])
        if c <= o:
            return False
        body = c - o
        if body <= 0:
            return False
        bottom_wick = o - l
        return (bottom_wick / body) < 0.05

    def _is_no_wick_bearish(self, row) -> bool:
        """
        Bearish candle (close < open) with NO top wick (or < 5% of body).
        Flat top = open is at or very near the high.
        """
        o = _as_float(row["open"])
        h = _as_float(row["high"])
        c = _as_float(row["close"])
        if c >= o:
            return False
        body = o - c
        if body <= 0:
            return False
        top_wick = h - o
        return (top_wick / body) < 0.05

    # ------------------------------------------------------------------
    # Rule 1 — Trend detection via BODY CLOSURES only (wicks ignored)
    # ------------------------------------------------------------------

    def _identify_trend(
        self, df: pd.DataFrame, lookback: int, pivot_strength: int
    ) -> str:
        """
        Transcript: 'You look at candle bodies. Ignore wicks completely.'

        Pivot high = close[i] strictly greater than all closes in
                     left window AND right window.
        Pivot low  = close[i] strictly less than all closes in both windows.

        Uptrend:   last two pivot highs rising AND last two pivot lows rising.
        Downtrend: last two pivot highs falling AND last two pivot lows falling.
        """
        n  = min(lookback, len(df))
        ps = pivot_strength
        if n < ps * 2 + 3:
            return "none"

        closes = df["close"].values[-n:]

        pivot_highs: List[float] = []
        pivot_lows:  List[float] = []

        for i in range(ps, len(closes) - ps):
            left  = closes[i - ps: i]
            right = closes[i + 1: i + ps + 1]
            if closes[i] > left.max() and closes[i] > right.max():
                pivot_highs.append(closes[i])
            if closes[i] < left.min() and closes[i] < right.min():
                pivot_lows.append(closes[i])

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return "none"

        h1, h2 = pivot_highs[-2], pivot_highs[-1]
        l1, l2 = pivot_lows[-2],  pivot_lows[-1]

        if h2 > h1 and l2 > l1:
            return "bullish"
        if h2 < h1 and l2 < l1:
            return "bearish"
        return "none"

    # ------------------------------------------------------------------
    # Rule 7 — Stop at WICK of structural pivot (doji filtered)
    # ------------------------------------------------------------------

    def _find_structure_stop(
        self, df: pd.DataFrame, side: str, pivot_strength: int
    ) -> float:
        """
        Transcript: 'Stop loss at the wick of the most recent higher low (buy)
                     or lower high (sell).'
                    'Doji candles don't count, we disregard them.'

        1. Scan backwards using CLOSE-based pivot detection (bodies only).
        2. Skip doji candles.
        3. Return the actual WICK (low for buys, high for sells) of the pivot.
        """
        closes = df["close"].values
        opens  = df["open"].values
        lows   = df["low"].values
        highs  = df["high"].values
        ps = pivot_strength

        if side == "buy":
            # Most recent pivot LOW by close, non-doji
            for i in range(len(closes) - ps - 1, ps - 1, -1):
                left  = closes[i - ps: i]
                right = closes[i + 1: i + ps + 1]
                if len(left) < ps or len(right) < ps:
                    continue
                if closes[i] < left.min() and closes[i] < right.min():
                    if not _is_doji(opens[i], highs[i], lows[i], closes[i]):
                        return float(lows[i])   # ← WICK low of this pivot bar
        else:
            # Most recent pivot HIGH by close, non-doji
            for i in range(len(closes) - ps - 1, ps - 1, -1):
                left  = closes[i - ps: i]
                right = closes[i + 1: i + ps + 1]
                if len(left) < ps or len(right) < ps:
                    continue
                if closes[i] > left.max() and closes[i] > right.max():
                    if not _is_doji(opens[i], highs[i], lows[i], closes[i]):
                        return float(highs[i])  # ← WICK high of this pivot bar

        # Fallback: recent extreme wick (should rarely hit)
        if side == "buy":
            return float(np.min(lows[-20:]))
        else:
            return float(np.max(highs[-20:]))

    # ------------------------------------------------------------------
    # Rule 3 — Session time filter (UTC)
    # ------------------------------------------------------------------

    def _is_restricted_time(self, ts_utc: pd.Timestamp, params: dict) -> bool:
        """
        Transcript rules:
          - Avoid first 3 hours of Asia open (default 23:00–02:00 UTC).
          - Avoid first 15-min candle of NY open (default 13:30 UTC).
          - Don't enter at/after NY close (default ≥ 21:00 UTC).
        """
        h = ts_utc.hour
        m = ts_utc.minute

        asia_open_utc  = int(params.get("asia_open_utc",   23))
        asia_avoid_hrs = int(params.get("asia_avoid_hours",  3))
        ny_close_utc   = int(params.get("ny_close_utc",    21))
        ny_open_utc    = int(params.get("ny_open_utc",     13))

        # Asia restricted window (wraps midnight)
        asia_restricted = {(asia_open_utc + d) % 24 for d in range(asia_avoid_hrs)}
        if h in asia_restricted:
            return True

        # NY open first candle (the :30 candle right after open)
        if h == ny_open_utc and m == 30:
            return True

        # At or after NY close — no new entries
        if h >= ny_close_utc:
            return True

        return False

    # ------------------------------------------------------------------
    # Rule 5 — Imbalance filter (wick-based zones + structure protection)
    # ------------------------------------------------------------------

    def _has_imbalance_against(
        self,
        df:             pd.DataFrame,
        side:           str,
        entry:          float,
        stop:           float,
        atr:            float,
        lookback:       int,
        threshold_mult: float,
        pivot_strength: int,
    ) -> bool:
        """
        Transcript Rule 5: 'No imbalances against your position.'

        An imbalance is a disproportionately large candle in the opposing
        direction whose zone overlaps the risk area between stop and entry.

        FIX vs previous version — three corrections:

        1. WICK-BASED ZONE: Imbalance zone uses the candle's full HIGH and LOW
           (wicks), not just the body open/close.
           Transcript: 'The way you draw imbalances is from wicks.'

        2. STRUCTURE PROTECTS: If a valid non-doji structural pivot exists
           BETWEEN the imbalance candle and current price, the imbalance is
           shielded and the trade CAN be taken.
           Transcript: 'If you have structure in between the imbalance and the
           compensation play you can disregard the imbalance.'

        3. SMALL IMBALANCE EXCEPTION: If the imbalance is small enough that
           the stop (already placed at the structural pivot wick) covers it
           entirely, we can take the trade.
           Transcript: 'If the imbalance is small and you can have stop loss
           below the imbalance without making it too big, that's fine.'
        """
        if atr <= 0 or len(df) < lookback + 2:
            return False

        threshold = threshold_mult * atr
        opens  = df["open"].values
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values

        zone_lo = min(entry, stop)
        zone_hi = max(entry, stop)
        start   = max(0, len(df) - lookback)
        ps      = pivot_strength

        for i in range(start, len(df) - 1):
            o, c = opens[i], closes[i]
            body = abs(c - o)
            if body < threshold:
                continue

            # Only care about candles moving AGAINST our trade
            if side == "buy"  and c >= o:
                continue   # need bearish candle against a buy
            if side == "sell" and c <= o:
                continue   # need bullish candle against a sell

            # FIX 1: Use WICK range for imbalance zone (not body range)
            imb_lo = lows[i]    # lowest wick of this imbalance candle
            imb_hi = highs[i]   # highest wick of this imbalance candle

            # Does this imbalance overlap the risk zone at all?
            if imb_hi < zone_lo or imb_lo > zone_hi:
                continue

            # Check if the imbalance has been filled by subsequent price action
            # Filled = price returned to the open (top of bearish body / bottom
            # of bullish body) — transcript: 'price came back and filled it'
            filled = False
            if side == "buy":
                # Bearish imbalance is filled when price comes back up to its open
                for j in range(i + 1, len(df)):
                    if highs[j] >= o:
                        filled = True
                        break
            else:
                # Bullish imbalance is filled when price comes back down to its open
                for j in range(i + 1, len(df)):
                    if lows[j] <= o:
                        filled = True
                        break

            if filled:
                continue  # filled — no longer a concern

            # FIX 2: STRUCTURE PROTECTS — check for a valid structural pivot
            # between this imbalance candle (index i) and the current bar.
            # Transcript: 'structure is a pullback and a re-break of highs/lows.
            #  One red candle in a green imbalance is enough structure.'
            protected = False
            for k in range(i + ps + 1, len(df) - ps):
                lk  = closes[k - ps: k]
                rk  = closes[k + 1: k + ps + 1]
                if len(lk) < ps or len(rk) < ps:
                    continue
                if _is_doji(opens[k], highs[k], lows[k], closes[k]):
                    continue
                if side == "buy":
                    # Need a higher pivot low after the bearish imbalance
                    if closes[k] < lk.min() and closes[k] < rk.min():
                        if lows[k] > lows[i]:   # higher low = structure formed
                            protected = True
                            break
                else:
                    # Need a lower pivot high after the bullish imbalance
                    if closes[k] > lk.max() and closes[k] > rk.max():
                        if highs[k] < highs[i]:  # lower high = structure formed
                            protected = True
                            break

            if protected:
                continue  # structure shields us from this imbalance

            # FIX 3: SMALL IMBALANCE EXCEPTION — if stop already covers it
            if side == "buy"  and stop <= imb_lo:
                continue   # stop is below the entire imbalance — we're fine
            if side == "sell" and stop >= imb_hi:
                continue   # stop is above the entire imbalance — we're fine

            # Unfilled, unprotected, not covered by stop → reject this trade
            return True

        return False

    # ------------------------------------------------------------------
    # Rule 6 — Near-miss / "almost tapped then reversed" invalidation
    # ------------------------------------------------------------------

    def _update_near_miss(
        self,
        signal:        NoWickCandle,
        current_high:  float,
        current_low:   float,
        breathing_room: float,
        df:            pd.DataFrame,
        pivot_strength: int,
    ) -> bool:
        """
        Transcript Rule 6: 'If price almost taps it, then reverses and hits
        your original TP — if it comes back and taps again, it's no longer
        valid. It has made the move you were trying to catch.'

        Step 1: Detect near-miss — price came within NEAR_MISS_BREATHING_MULT
                * breathing_room of the tap level.
        Step 2: At near-miss moment, estimate what TP would have been
                (entry ± risk) using the current structural stop.
        Step 3: On subsequent bars, if price reaches that TP level, the signal
                is dead — return True (= invalidate this signal).

        Returns True if signal should be invalidated/dropped.
        """
        tap = signal.price

        # Step 1: detect near-miss if not already seen
        if not signal.near_miss:
            threshold = breathing_room * NEAR_MISS_BREATHING_MULT
            near = (
                (signal.side == "buy"  and current_low  <= tap + threshold and current_low  > tap) or
                (signal.side == "sell" and current_high >= tap - threshold and current_high < tap)
            )
            if near:
                # Estimate TP at this moment
                stop_base = self._find_structure_stop(df, signal.side, pivot_strength)
                if signal.side == "buy":
                    stop = stop_base - breathing_room
                    if stop < tap:
                        risk = tap - stop
                        signal.near_miss_tp = tap + risk    # TP is above entry
                else:
                    stop = stop_base + breathing_room
                    if stop > tap:
                        risk = stop - tap
                        signal.near_miss_tp = tap - risk    # TP is below entry
                signal.near_miss = True
            return False  # haven't confirmed invalidation yet

        # Step 2: near-miss already detected — has price hit the estimated TP?
        if signal.near_miss_tp is None:
            return False  # couldn't compute TP at near-miss time — keep signal

        if signal.side == "buy"  and current_high >= signal.near_miss_tp:
            return True   # TP level hit without us — signal is dead
        if signal.side == "sell" and current_low  <= signal.near_miss_tp:
            return True   # TP level hit without us — signal is dead

        return False

    # ------------------------------------------------------------------
    # Trade management signal detection
    # ------------------------------------------------------------------

    def _get_management_signals(
        self, df: pd.DataFrame, open_side: str, pivot_strength: int
    ) -> List[dict]:
        """
        Transcript: 'If there's a no-wick candle against your open position,
        wait for it to tap. If in profit: move SL to breakeven. If in drawdown:
        move TP to breakeven.'

        Scans the most recent bars for no-wick candles against the current
        open position. Returns a list of management event dicts that the
        broker/execution engine should act on.

        The caller is responsible for:
          - Checking if current price taps any of these levels.
          - Moving SL or TP to breakeven accordingly.
        """
        signals = []
        # Only look at the last 9 bars (max tap window)
        lookback = min(9, len(df))
        for i in range(len(df) - lookback, len(df)):
            row = df.iloc[i]
            o = _as_float(row["open"])
            h = _as_float(row["high"])
            c = _as_float(row["close"])
            l = _as_float(row["low"])
            if open_side == "buy" and self._is_no_wick_bearish(row):
                signals.append({
                    "type":       "no_wick_against",
                    "level":      o,           # tap this level → move to BE
                    "direction":  "bearish",
                    "bar_index":  i,
                })
            elif open_side == "sell" and self._is_no_wick_bullish(row):
                signals.append({
                    "type":       "no_wick_against",
                    "level":      o,
                    "direction":  "bullish",
                    "bar_index":  i,
                })
        return signals

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:

        # ---- pair setup ----
        pair     = symbol.replace("/", "").upper()
        pip_size = 0.01 if "JPY" in pair else 0.0001

        if pair not in SUPPORTED_PAIRS:
            return Order(symbol, None, None, None, None,
                         f"unsupported pair {pair}", {})

        breathing_room = PIP_MAP[pair] * pip_size

        # ---- params ----
        max_candles_to_tap = int(self.params.get("max_candles_to_tap",   9))
        max_trades_per_day = int(self.params.get("max_trades_per_day",   5))
        trend_lookback     = int(self.params.get("trend_lookback",      50))
        pivot_strength     = int(self.params.get("pivot_strength",       2))
        imbalance_lookback = int(self.params.get("imbalance_lookback",  40))
        imbalance_mult     = float(self.params.get("imbalance_atr_mult", 1.5))
        bar_minutes        = int(self.params.get("bar_minutes",         15))

        max_tap_td = pd.Timedelta(minutes=bar_minutes * (max_candles_to_tap + 1))

        # ---- warmup guard ----
        warmup = trend_lookback + pivot_strength * 2 + 5
        if df is None or len(df) < warmup:
            return Order(symbol, None, None, None, None, "warmup", {})

        # ---- timestamps ----
        idx_utc  = _to_utc(df.index)
        last_utc = idx_utc[-1]

        # ---- day tracking ----
        day_utc = last_utc.normalize()
        st = self._get_state(symbol)
        if st.day is None or st.day != day_utc:
            st.day         = day_utc
            st.trades_today = 0
            # Preserve signals still within their tap window across midnight
            st.pending_signals = [
                s for s in st.pending_signals
                if (last_utc - s.created_time) <= max_tap_td
            ]

        if st.trades_today >= max_trades_per_day:
            return Order(symbol, None, None, None, None, "max trades/day", {})

        # ---- Rule 3: session filter ----
        if self._is_restricted_time(last_utc, self.params):
            return Order(symbol, None, None, None, None, "restricted session", {})

        # ---- Rule 1: trend via body closures ----
        trend = self._identify_trend(df, trend_lookback, pivot_strength)
        if trend == "none":
            return Order(symbol, None, None, None, None, "no clear trend", {})

        current_row  = df.iloc[-1]
        current_low  = _as_float(current_row["low"])
        current_high = _as_float(current_row["high"])
        atr          = _simple_atr(df, 14)

        # ---- Register new no-wick signal on this bar ----
        # (Only with the trend per Rule 1)
        if trend == "bullish" and self._is_no_wick_bullish(current_row):
            st.pending_signals.append(NoWickCandle(
                created_time=last_utc,
                price=_as_float(current_row["open"]),
                side="buy",
            ))

        if trend == "bearish" and self._is_no_wick_bearish(current_row):
            st.pending_signals.append(NoWickCandle(
                created_time=last_utc,
                price=_as_float(current_row["open"]),
                side="sell",
            ))

        # ---- Process pending signals ----
        still_pending:   List[NoWickCandle] = []
        order_to_return: Optional[Order]    = None

        for signal in st.pending_signals:

            # Rule 4: elapsed measured by timestamp (not context-window index)
            elapsed_min     = (last_utc - signal.created_time).total_seconds() / 60
            candles_elapsed = int(elapsed_min / bar_minutes)

            # Expire signals older than max_candles_to_tap
            if candles_elapsed > max_candles_to_tap:
                continue  # drop

            # Don't check tap on the same bar the signal was created
            if candles_elapsed == 0:
                still_pending.append(signal)
                continue

            # Rule 6: update near-miss state; drop if TP was hit without us
            if self._update_near_miss(
                signal, current_high, current_low,
                breathing_room, df, pivot_strength
            ):
                continue  # signal is dead — price already made the move

            # Check for tap
            tapped = (
                (signal.side == "buy"  and current_low  <= signal.price) or
                (signal.side == "sell" and current_high >= signal.price)
            )

            if not tapped:
                still_pending.append(signal)
                continue

            # Already found a valid entry this bar — keep remaining for next bar
            if order_to_return is not None:
                still_pending.append(signal)
                continue

            # ---- Rule 7: stop at wick of structural pivot ----
            stop_base = self._find_structure_stop(df, signal.side, pivot_strength)

            if signal.side == "buy":
                stop = stop_base - breathing_room
                if stop >= signal.price:
                    continue   # invalid geometry
                risk = signal.price - stop
                take = signal.price + risk

            else:
                stop = stop_base + breathing_room
                if stop <= signal.price:
                    continue
                risk = stop - signal.price
                take = signal.price - risk

            # ---- Rule 5: imbalance filter (wick-based + structure exception) ----
            if self._has_imbalance_against(
                df, signal.side, signal.price, stop,
                atr, imbalance_lookback, imbalance_mult, pivot_strength,
            ):
                continue  # unfilled imbalance against us — skip

            # ---- Build management signals for execution engine ----
            mgmt = self._get_management_signals(df, signal.side, pivot_strength)

            # ---- All checks passed — generate order ----
            st.trades_today += 1
            order_to_return = Order(
                symbol=symbol,
                side=signal.side,
                entry=signal.price,
                stop=stop,
                take=take,
                reason=(
                    f"No-wick {signal.side} | {candles_elapsed} bar(s) | "
                    f"trend={trend} | risk={risk:.5f} | "
                    f"br={breathing_room:.5f}"
                ),
                meta={
                    "candles_to_tap":      candles_elapsed,
                    "breathing_room":      breathing_room,
                    "trend":               trend,
                    "trail_mode":          "fixed",    # 1:1 fixed — no trailing
                    "atr":                 atr,
                    "management_signals":  mgmt,       # Rule trade-mgmt: broker acts on these
                    # Rule 9 flag: caller should check if this is also a 30min no-wick
                    # and prefer 30min version if so. This strategy runs on 15min only.
                    "htf_priority_note":   "check_30min",
                    # Rule 2: news filter must be applied externally
                    "news_filter_note":    "check_forexfactory_red_folders",
                },
            )
            # signal consumed — do NOT re-add to still_pending

        st.pending_signals = still_pending

        if order_to_return is not None:
            return order_to_return

        return Order(symbol, None, None, None, None, "waiting for tap", {})
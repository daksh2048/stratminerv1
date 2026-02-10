from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import csv
import numpy as np
import pandas as pd

from src.core.types import Order

Side = str


def _to_float(x) -> float:
    """Robustly convert pandas / numpy values to a plain Python float."""
    if isinstance(x, (pd.Series, pd.Index)):
        if len(x) == 0:
            return float("nan")
        return float(x.iloc[0])
    if isinstance(x, np.generic):
        return float(x)
    return float(x)


@dataclass
class Position:
    symbol: str
    side: Side
    size: float
    entry: float
    stop: float
    take: float
    open_time: Any
    strategy: str
    meta: Dict[str, Any] = field(default_factory=dict)

    close_time: Optional[Any] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: Optional[str] = None  # "tp", "stopped", "partial_tp", "data_end"


class PaperBroker:
    def __init__(
        self,
        starting_balance: float,
        risk_per_trade: float,
        slippage_bps: float,
        trades_csv: str = "trades.csv",
        max_loss_pct: float = 0.03,
    ):
        self.balance = float(starting_balance)
        self.initial_balance = float(starting_balance)
        self.max_loss_pct = float(max_loss_pct)
        self.trading_enabled = True

        self.risk_per_trade = float(risk_per_trade)
        self.slippage_bps = float(slippage_bps)
        self.trades_csv = trades_csv

        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self._candle_buffer: Dict[str, list] = {}  # symbol -> last 50 candles for hybrid trailing

    def _breached_max_loss(self) -> bool:
        return self.balance <= self.initial_balance * (1.0 - self.max_loss_pct)

    # ---------- internal helpers ----------

    def _apply_slippage(self, price: float, side: Side) -> float:
        """Apply slippage in bps to a price."""
        if self.slippage_bps == 0:
            return float(price)
        factor = self.slippage_bps / 10_000.0
        if side == "buy":
            return float(price) * (1 + factor)
        else:
            return float(price) * (1 - factor)

    def _log_closed_position(self, pos: Position) -> None:
        # FIX: Use initial_stop for accurate R calculation
        initial_stop = float(pos.meta.get("initial_stop", pos.stop))
        risk_dollars = abs(pos.entry - initial_stop) * pos.size
        R = (pos.pnl / risk_dollars) if (risk_dollars and risk_dollars > 0) else 0.0

        if pos.pnl > 0:
            result = "win"
        elif pos.pnl < 0:
            result = "loss"
        else:
            result = "breakeven"

        with open(self.trades_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    pos.symbol,
                    pos.side,
                    pos.size,
                    pos.entry,
                    pos.stop,
                    pos.take,
                    pos.open_time,
                    pos.close_time,
                    pos.exit_price,
                    pos.pnl,  # $
                    R,        # R-multiple
                    result,
                    pos.strategy,
                    pos.status,
                    self.balance,
                ]
            )

    def _close_position(self, pos: Position, now: Any, exit_price: float, status: str) -> None:
        """Close a position completely at exit_price (with slippage applied)."""
        exit_side = "sell" if pos.side == "buy" else "buy"
        exit_price_slipped = self._apply_slippage(float(exit_price), exit_side)

        pos.close_time = now
        pos.exit_price = exit_price_slipped
        pos.status = status

        if pos.side == "buy":
            pos.pnl = (pos.exit_price - pos.entry) * pos.size
        else:
            pos.pnl = (pos.entry - pos.exit_price) * pos.size

        self.balance += pos.pnl
        self.closed_positions.append(pos)
        self._log_closed_position(pos)

        if pos in self.open_positions:
            self.open_positions.remove(pos)

    def _do_partial_take(self, pos: Position, now: Any, take_price: float, pct: float) -> None:
        """
        Realize a partial take on pos at take_price (with slippage).
        Leaves remaining position open and updates meta accordingly.
        """
        meta = pos.meta or {}
        pct = float(pct)

        closed_size = pos.size * pct
        remaining = pos.size - closed_size
        if closed_size <= 0 or remaining <= 0:
            # nothing meaningful to do -> treat as full close elsewhere
            return

        exit_side = "sell" if pos.side == "buy" else "buy"
        exit_price_slipped = self._apply_slippage(float(take_price), exit_side)

        # realized pnl on partial
        if pos.side == "buy":
            pnl = (exit_price_slipped - pos.entry) * closed_size
        else:
            pnl = (pos.entry - exit_price_slipped) * closed_size

        self.balance += pnl

        # log "partial" line as separate closed Position
        part = Position(
            symbol=pos.symbol,
            side=pos.side,
            size=closed_size,
            entry=pos.entry,
            stop=pos.stop,
            take=pos.take,
            open_time=pos.open_time,
            strategy=pos.strategy,
            meta=dict(meta),
        )
        part.close_time = now
        part.exit_price = exit_price_slipped
        part.pnl = pnl
        part.status = "partial_tp"
        self.closed_positions.append(part)
        self._log_closed_position(part)

        # keep runner open
        pos.size = remaining
        meta["partial_done"] = True

        # Move stop to BE after partial
        if bool(meta.get("move_stop_to_be", False)):
            if pos.side == "buy":
                pos.stop = max(pos.stop, pos.entry)
            else:
                pos.stop = min(pos.stop, pos.entry)

        # Set runner take or disable it
        mode = str(meta.get("runner_take_mode", "rr")).lower()
        if mode == "rr":
            rr = float(meta.get("runner_rr", 3.0))
            initial_stop = float(meta.get("initial_stop", pos.stop))
            risk = abs(pos.entry - initial_stop)
            if risk <= 0:
                pos.take = float("nan")
            else:
                if pos.side == "buy":
                    pos.take = pos.entry + rr * risk
                else:
                    pos.take = pos.entry - rr * risk
        else:
            pos.take = float("nan")  # rely on trailing/stop

    def _event_sequence(self, pos_side: Side, open_: float, close: float) -> List[str]:
        """
        Intrabar path heuristic:
          - Green candle (close >= open): assume LOW occurs before HIGH.
          - Red candle (close < open): assume HIGH occurs before LOW.

        For LONG: stop is LOW-side, take is HIGH-side.
        For SHORT: stop is HIGH-side, take is LOW-side.

        Returns event order list among ["stop","take"].
        """
        green = close >= open_
        if pos_side == "buy":
            return ["stop", "take"] if green else ["take", "stop"]
        else:  # sell
            return ["take", "stop"] if green else ["stop", "take"]

    # ---------- public API ----------

    def open_from_order(
        self,
        order: Order,
        strategy: str,
        now: Any,
        market_close_price: float,
    ) -> Optional[Position]:
        """
        Create a Position from a strategy Order and add it to open positions.
        Size is based on risk_per_trade * current balance.
        Limits to one open position per (symbol, strategy).
        
        FIXED: Now applies slippage BEFORE calculating position size.
        """
        # kill-switch
        if (not self.trading_enabled) or self._breached_max_loss():
            return None

        # sanity checks
        if order.side not in ("buy", "sell"):
            return None
        if order.entry is None or order.stop is None or order.take is None:
            return None

        # don't stack same symbol+strategy
        for pos in self.open_positions:
            if pos.symbol == order.symbol and pos.strategy == strategy:
                return None

        side: Side = order.side  # type: ignore

        entry_price = float(market_close_price)
        stop_price = float(order.stop)
        take_price = float(order.take)

        # FIX: Apply slippage FIRST, before calculating size
        entry_price_slipped = self._apply_slippage(entry_price, side)

        # FIX: Validate stop/take make sense relative to entry
        if side == "buy":
            if stop_price >= entry_price_slipped or take_price <= entry_price_slipped:
                return None  # Invalid long setup
        else:
            if stop_price <= entry_price_slipped or take_price >= entry_price_slipped:
                return None  # Invalid short setup

        # FIX: Now calculate size with CORRECT slipped entry
        risk_per_unit = (entry_price_slipped - stop_price) if side == "buy" else (stop_price - entry_price_slipped)
        if risk_per_unit <= 0:
            return None

        risk_amount = self.balance * self.risk_per_trade
        size = risk_amount / risk_per_unit

        pos = Position(
            symbol=order.symbol,
            side=side,
            size=size,
            entry=entry_price_slipped,
            stop=stop_price,
            take=take_price,
            open_time=now,
            strategy=strategy,
            meta=order.meta.copy() if order.meta else {},
        )

        # defaults / anchors
        pos.meta.setdefault("initial_stop", stop_price)
        pos.meta.setdefault("initial_take", take_price)
        pos.meta.setdefault("partial_done", False)
        pos.meta.setdefault("trail_active", False)

        if side == "buy":
            pos.meta.setdefault("peak", entry_price_slipped)
        else:
            pos.meta.setdefault("trough", entry_price_slipped)

        self.open_positions.append(pos)
        return pos

    def update_with_candle(self, candle: pd.Series, now: Any, symbol: str, tf: str):
        # Maintain rolling candle buffer for hybrid trailing (last 50 bars)
        if symbol not in self._candle_buffer:
            self._candle_buffer[symbol] = []
        self._candle_buffer[symbol].append(candle)
        if len(self._candle_buffer[symbol]) > 50:
            self._candle_buffer[symbol].pop(0)

        # include open for gap logic; fallback to close if missing
        close = _to_float(candle["close"])
        open_ = _to_float(candle["open"]) if "open" in candle else close
        high = _to_float(candle["high"])
        low = _to_float(candle["low"])

        closed_positions: List[Position] = []

        for pos in list(self.open_positions):
            # only positions for this symbol + tf
            if pos.symbol != symbol:
                continue
            pos_tf = (pos.meta or {}).get("tf")
            if pos_tf is not None and pos_tf != tf:
                continue

            meta = pos.meta or {}

            # ----------------------------
            # 0) Position expiry via max_bars_open
            #    Strategies can pass max_bars_open in meta to enforce
            #    a time-based close (e.g. EOD = bars until session close).
            #    This prevents multi-day holds on intraday strategies.
            # ----------------------------
            max_bars = meta.get("max_bars_open", None)
            if max_bars is not None:
                bars_open = int(meta.get("bars_open", 0)) + 1
                meta["bars_open"] = bars_open
                if bars_open >= int(max_bars):
                    self._close_position(pos, now, close, "max_bars")
                    closed_positions.append(pos)
                    continue

            # ----------------------------
            # 1) Trailing stop maintenance (with advanced mode support)
            # ----------------------------
            trail_mode = str(meta.get("trail_mode", "percent"))
            activate_after_partial = bool(meta.get("trail_activate_after_partial", False))
            partial_done = bool(meta.get("partial_done", False))
            
            # Activate trailing
            trail_pct = meta.get("trail_pct", None)
            if trail_pct is not None or trail_mode != "percent":
                if (not activate_after_partial) or partial_done:
                    meta["trail_active"] = True
            
            # Execute trailing if active
            if bool(meta.get("trail_active", False)):
                # Try to import advanced trailing module
                try:
                    from src.core.trailing_stops import calculate_trailing_stop
                    use_advanced = True
                except ImportError:
                    use_advanced = False
                
                # Advanced trailing (if module available and mode is not "percent")
                if use_advanced and trail_mode != "percent":
                    # Build df_recent from live candle buffer (used by chandelier, hybrid,
                    # structure modes AND for live ATR recalculation).
                    df_recent = None
                    if symbol in self._candle_buffer and len(self._candle_buffer[symbol]) >= 5:
                        df_recent = pd.DataFrame(self._candle_buffer[symbol])

                    # Recalculate ATR from live buffer so trailing adapts to current
                    # volatility instead of being frozen at entry-time ATR.
                    atr_current = None
                    if trail_mode in ("atr", "chandelier", "hybrid"):
                        if df_recent is not None and len(df_recent) >= 15:
                            try:
                                from src.core.trailing_stops import calculate_atr
                                atr_current = calculate_atr(df_recent, period=14)
                            except Exception:
                                pass
                        # Fallback: use entry-time ATR stored in meta if buffer not warm
                        if atr_current is None:
                            _meta_atr = meta.get("atr", None)
                            if _meta_atr is not None:
                                atr_current = float(_meta_atr)

                    # Call advanced trailing
                    new_stop = calculate_trailing_stop(
                        pos_side=pos.side,
                        pos_entry=pos.entry,
                        current_stop=pos.stop,
                        candle_high=high,
                        candle_low=low,
                        meta=meta,
                        df_recent=df_recent,
                        atr_current=atr_current,
                    )
                    
                    # Apply new stop
                    if new_stop is not None:
                        if pos.side == "buy" and new_stop > pos.stop:
                            pos.stop = new_stop
                        elif pos.side == "sell" and new_stop < pos.stop:
                            pos.stop = new_stop
                
                # Simple percentage trailing (fallback or if mode is "percent")
                elif trail_pct is not None and float(trail_pct) > 0:
                    tpct = float(trail_pct)
                    if pos.side == "buy":
                        current_peak = float(meta.get("peak", pos.entry))
                        meta["peak"] = max(current_peak, high)
                        peak = float(meta["peak"])
                        new_stop = peak * (1.0 - tpct)
                        if new_stop > pos.stop:
                            pos.stop = new_stop
                    else:
                        current_trough = float(meta.get("trough", pos.entry))
                        meta["trough"] = min(current_trough, low)
                        trough = float(meta["trough"])
                        new_stop = trough * (1.0 + tpct)
                        if new_stop < pos.stop:
                            pos.stop = new_stop

            # Move stop to breakeven after partial (redundant safety; also done in partial function)
            if partial_done and bool(meta.get("move_stop_to_be", False)):
                if pos.side == "buy":
                    pos.stop = max(pos.stop, pos.entry)
                else:
                    pos.stop = min(pos.stop, pos.entry)

            # ----------------------------
            # 2) Hit detection (gap-aware)
            # ----------------------------
            take_is_valid = not pd.isna(pos.take)

            if pos.side == "buy":
                stop_gap = open_ <= pos.stop
                take_gap = take_is_valid and (open_ >= pos.take)

                stop_hit = stop_gap or (low <= pos.stop)
                take_hit = take_gap or (take_is_valid and (high >= pos.take))
            else:
                stop_gap = open_ >= pos.stop
                take_gap = take_is_valid and (open_ <= pos.take)

                stop_hit = stop_gap or (high >= pos.stop)
                take_hit = take_gap or (take_is_valid and (low <= pos.take))

            if (not stop_hit) and (not take_hit):
                continue

            # ----------------------------
            # 3) Resolve event order fairly
            # ----------------------------
            events: List[str] = []
            # gap is first if it happens
            if stop_gap:
                events.append("stop")
            elif take_gap:
                events.append("take")

            # add intrabar sequence for remaining hits
            seq = self._event_sequence(pos.side, open_, close)
            for e in seq:
                if e == "stop" and stop_hit and ("stop" not in events):
                    events.append("stop")
                if e == "take" and take_hit and ("take" not in events):
                    events.append("take")

            # ----------------------------
            # 4) Execute events in order
            # ----------------------------
            for e in events:
                # if position already closed by previous event, break
                if pos not in self.open_positions:
                    break

                # STOP
                if e == "stop":
                    exit_price = open_ if stop_gap else pos.stop
                    self._close_position(pos, now, exit_price, "stopped")
                    closed_positions.append(pos)
                    break  # fully closed

                # TAKE (partial/full)
                if e == "take":
                    exit_price = open_ if take_gap else pos.take

                    pt = float(meta.get("partial_take_pct", 1.0))
                    partial_done_now = bool(meta.get("partial_done", False))

                    # partial take
                    if (pt < 1.0) and (not partial_done_now):
                        self._do_partial_take(pos, now, exit_price, pt)
                        # after partial, we do NOT force-close in the same candle automatically;
                        # BUT if stop is also in the events list later, it will close runner.
                        continue

                    # full take (or runner TP)
                    self._close_position(pos, now, exit_price, "tp")
                    closed_positions.append(pos)
                    break

        # kill-switch after updates
        if self._breached_max_loss():
            self.trading_enabled = False

        return closed_positions

    def force_close_all(
        self,
        now: Any,
        market_close_price: float,
        symbol: str,
        tf: str,
        status: str = "data_end",
    ):
        """Close remaining positions for this (symbol, tf) at the last close."""
        for pos in list(self.open_positions):
            if pos.symbol != symbol:
                continue
            pos_tf = (pos.meta or {}).get("tf")
            if pos_tf is not None and pos_tf != tf:
                continue

            self._close_position(pos, now, float(market_close_price), status)
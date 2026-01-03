from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.core.types import Order
Side = str


def _to_float(x) -> float:
    """
    Robustly convert pandas / numpy values to a plain Python float.
    This avoids the 'truth value of a Series is ambiguous' issue.
    """
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
    open_time: Any   # typically a pandas.Timestamp
    strategy: str
    meta: Dict[str, Any] = field(default_factory=dict)

    # Filled when trade is closed
    close_time: Optional[Any] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: Optional[str] = None  # "tp", "stopped", "manual"


class PaperBroker:
    def __init__(self, starting_balance: float, risk_per_trade: float, slippage_bps: float,
                trades_csv: str = "trades.csv", max_loss_pct: float = 0.03):
        self.balance = float(starting_balance)
        self.initial_balance = float(starting_balance)
        self.max_loss_pct = float(max_loss_pct)
        self.trading_enabled = True

        self.risk_per_trade = float(risk_per_trade)
        self.slippage_bps = float(slippage_bps)
        self.trades_csv = trades_csv
        self.open_positions = []
        self.closed_positions = []

    def _breached_max_loss(self) -> bool:
        return self.balance <= self.initial_balance * (1.0 - self.max_loss_pct)

    def open_from_order(self, order: Order, strategy: str, now: Any, market_close_price: float):
            # kill-switch: stop opening new trades after max loss
        if (not self.trading_enabled) or self._breached_max_loss():
            return None

    # ---------- internal helpers ----------

    def _apply_slippage(self, price: float, side: Side) -> float:
        """
        Apply slippage in basis points (bps) to a price.
        Example: 1.5 bps = 0.00015 = 0.015%
        """
        if self.slippage_bps == 0:
            return price
        factor = self.slippage_bps / 10_000.0
        if side == "buy":
            return price * (1 + factor)
        else:  # sell
            return price * (1 - factor)

    def _log_closed_position(self, pos: Position) -> None:
        # Compute risk in dollars (for R-multiple)
        risk_dollars = abs(pos.entry - pos.stop) * pos.size
        if risk_dollars > 0:
            R = pos.pnl / risk_dollars
        else:
            R = 0.0

        # Classify trade as win / loss / breakeven
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
                    pos.pnl,     # dollar profit/loss
                    R,           # R-multiple
                    result,      # win / loss / breakeven
                    pos.strategy,
                    pos.status,
                    self.balance,
                ]
            )

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
        Position size is based on risk_per_trade * current balance.
        Also limits to one open position per (symbol, strategy).
        """
        # Basic sanity checks
        if order.side not in ("buy", "sell"):
            return None
        if order.entry is None or order.stop is None or order.take is None:
            return None

        # NEW: don't stack multiple positions for same symbol+strategy
        for pos in self.open_positions:
            if pos.symbol == order.symbol and pos.strategy == strategy:
                # Already in a trade for this symbol+strategy, skip opening another
                return None

        side: Side = order.side  # type: ignore
        entry_price = float(market_close_price)
        stop_price = float(order.stop)
        take_price = float(order.take)

        # Risk per unit (distance between entry and stop)
        if side == "buy":
            risk_per_unit = entry_price - stop_price
        else:  # sell
            risk_per_unit = stop_price - entry_price

        if risk_per_unit <= 0:
            # Invalid risk, don't open the trade
            return None

        # Risk we want to take in dollars on this trade
        risk_amount = self.balance * self.risk_per_trade
        size = risk_amount / risk_per_unit

        # Apply slippage to entry
        entry_price_slipped = self._apply_slippage(entry_price, side)

        pos = Position(
            symbol=order.symbol,
            side=side,
            size=size,
            entry=entry_price_slipped,
            stop=stop_price,
            take=take_price,
            open_time=now,
            strategy=strategy,
            meta=order.meta.copy(),
        )
        self.open_positions.append(pos)
        return pos

    def update_with_candle(self, candle: pd.Series, now: Any, symbol: str, tf: str):
        high = _to_float(candle["high"])
        low  = _to_float(candle["low"])
        close = _to_float(candle["close"])

        closed_positions = []

        for pos in list(self.open_positions):

            # ✅ only update positions that belong to THIS symbol + tf
            if pos.symbol != symbol:
                continue
            pos_tf = (pos.meta or {}).get("tf")
            if pos_tf is not None and pos_tf != tf:
                continue

            stop_hit = (low <= pos.stop <= high)
            take_hit = (low <= pos.take <= high)

            exit_price = None
            status = None

            # stop-first rule
            if stop_hit:
                exit_price = pos.stop
                status = "stopped"
            elif take_hit:
                exit_price = pos.take
                status = "tp"

            if exit_price is not None:
                exit_price_slipped = self._apply_slippage(
                    exit_price, "sell" if pos.side == "buy" else "buy"
                )

                pos.close_time = now
                pos.exit_price = exit_price_slipped
                pos.status = status

                if pos.side == "buy":
                    pos.pnl = (pos.exit_price - pos.entry) * pos.size
                else:
                    pos.pnl = (pos.entry - pos.exit_price) * pos.size

                self.balance += pos.pnl
                closed_positions.append(pos)
                self.closed_positions.append(pos)

                self.open_positions.remove(pos)
                self._log_closed_position(pos)

        # optional: if max loss breached after closes, halt further trading
        if self._breached_max_loss():
            self.trading_enabled = False

        return closed_positions

    def force_close_all(self, now: Any, market_close_price: float, symbol: str, tf: str, status: str = "data_end"):
    # close remaining positions for this (symbol, tf) at the last close
        for pos in list(self.open_positions):
            if pos.symbol != symbol:
                continue
            pos_tf = (pos.meta or {}).get("tf")
            if pos_tf is not None and pos_tf != tf:
                continue

            exit_price_slipped = self._apply_slippage(
                float(market_close_price), "sell" if pos.side == "buy" else "buy"
            )

            pos.close_time = now
            pos.exit_price = exit_price_slipped
            pos.status = status

            if pos.side == "buy":
                pos.pnl = (pos.exit_price - pos.entry) * pos.size
            else:
                pos.pnl = (pos.entry - pos.exit_price) * pos.size

            self.balance += pos.pnl
            self.closed_positions.append(pos)

            self.open_positions.remove(pos)
            self._log_closed_position(pos)
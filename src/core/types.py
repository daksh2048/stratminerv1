from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Order:
    """
    A signal from a strategy: either a real order or 'no setup'.

    side: 'buy', 'sell', or None
    entry/stop/take: prices (floats) or None if no trade
    reason: human-readable explanation of the signal
    meta: extra info (recent high/low, indicators, etc.)
    """
    symbol: str
    side: Optional[str]
    entry: Optional[float]
    stop: Optional[float]
    take: Optional[float]
    reason: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """
    A simulated open or closed position in the paper account.
    """
    symbol: str
    side: str                # 'buy' or 'sell'
    size: float              # positive quantity (units/contracts)
    entry: float
    stop: float
    take: float

    open_time: Any           # e.g. pandas.Timestamp
    close_time: Optional[Any] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    strategy: str = ""       # which strategy opened this position
    status: str = "open"     # 'open', 'stopped', 'tp', 'closed'
    meta: Dict[str, Any] = field(default_factory=dict)  # Strategy metadata (trailing stops, etc.)
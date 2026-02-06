from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import yfinance as yf


Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "1d"]


def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make yfinance output always have columns:
      open, high, low, close, volume

    Handles:
      - Single-index columns: Open/High/Low/Close/Volume
      - MultiIndex columns: (Open, SPY) or (SPY, Open) or other variations
    """
    if df is None or df.empty:
        return df

    cols = df.columns

    # ---- MultiIndex handling ----
    if isinstance(cols, pd.MultiIndex):
        # We'll try to find which level contains OHLCV labels.
        # Common patterns:
        #   level0: Open/High/... , level1: SPY
        #   level0: SPY, level1: Open/High/...
        level0 = [str(x).strip().lower() for x in cols.get_level_values(0)]
        level1 = [str(x).strip().lower() for x in cols.get_level_values(1)]

        ohlcv = {"open", "high", "low", "close", "volume", "adj close"}

        level0_has_ohlc = any(x in ohlcv for x in level0)
        level1_has_ohlc = any(x in ohlcv for x in level1)

        if level0_has_ohlc and not level1_has_ohlc:
            # keep only the OHLCV columns from level0; ignore ticker level1
            # Build a mapping from full multiindex -> level0 label
            new_cols = [str(a).strip() for (a, b) in cols]
            df = df.copy()
            df.columns = new_cols  # e.g., "Open", "High", ...
        elif level1_has_ohlc and not level0_has_ohlc:
            # OHLCV is level1
            new_cols = [str(b).strip() for (a, b) in cols]
            df = df.copy()
            df.columns = new_cols
        else:
            # fallback: join both levels, then we'll match case-insensitively
            new_cols = [f"{a}_{b}" for (a, b) in cols]
            df = df.copy()
            df.columns = new_cols

    # ---- Single-index normalize (case-insensitive) ----
    # Make a lookup from lowercase col -> original col name
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            key = cand.strip().lower()
            if key in lower_map:
                return lower_map[key]
        return None

    open_col = pick("open", "Open")
    high_col = pick("high", "High")
    low_col = pick("low", "Low")
    close_col = pick("close", "Close")
    vol_col = pick("volume", "Volume")

    # Some fallbacks if columns got joined like "Open_SPY"
    if open_col is None:
        open_col = next((c for c in df.columns if str(c).lower().endswith("_open")), None) or \
                   next((c for c in df.columns if str(c).lower().startswith("open_")), None)
    if high_col is None:
        high_col = next((c for c in df.columns if str(c).lower().endswith("_high")), None) or \
                   next((c for c in df.columns if str(c).lower().startswith("high_")), None)
    if low_col is None:
        low_col = next((c for c in df.columns if str(c).lower().endswith("_low")), None) or \
                  next((c for c in df.columns if str(c).lower().startswith("low_")), None)
    if close_col is None:
        close_col = next((c for c in df.columns if str(c).lower().endswith("_close")), None) or \
                    next((c for c in df.columns if str(c).lower().startswith("close_")), None)
    if vol_col is None:
        vol_col = next((c for c in df.columns if str(c).lower().endswith("_volume")), None) or \
                  next((c for c in df.columns if str(c).lower().startswith("volume_")), None)

    if any(x is None for x in [open_col, high_col, low_col, close_col, vol_col]):
        # show what we actually got so debugging is easy
        got = list(map(str, df.columns))
        raise RuntimeError(
            f"yfinance output columns not recognized. "
            f"Need OHLCV but got columns={got[:25]}{'...' if len(got) > 25 else ''}"
        )

    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[vol_col], errors="coerce"),
        },
        index=df.index,
    ).dropna()

    out.index.name = "ts"
    return out


@dataclass
class CandleFeed:
    exchange: str
    symbol: str
    timeframe: Timeframe = "5m"

    def _yf_interval(self) -> str:
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "1d": "1d",
        }
        if self.timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
        return mapping[self.timeframe]

    def fetch(self, period: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        interval = self._yf_interval()

        # defaults
        if period is None:
            if interval == "1m":
                period = "7d"
            elif interval in ("5m", "15m", "30m", "60m"):
                period = "60d"
            else:
                period = "5y"

        data = yf.download(
            self.symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=True,
        )

        if data is None or data.empty:
            raise RuntimeError(f"No data returned for {self.symbol} interval={interval} period={period}")

        df = _normalize_yf_columns(data)

        if limit is not None:
            df = df.tail(int(limit)).copy()

        return df

    def fetch_latest(self, limit: int = 500) -> pd.DataFrame:
        return self.fetch(period=None, limit=limit)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yfinance as yf


Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "1d"]


@dataclass
class CandleFeed:
    """
    Fetch OHLCV candles for stocks/ETFs using Yahoo Finance (yfinance).

    This replaces the old OANDA/ccxt-based feed. It returns a DataFrame with:
    index: timestamp (ts)
    columns: open, high, low, close, volume
    """

    exchange: str
    symbol: str
    timeframe: Timeframe = "5m"

    def __post_init__(self) -> None:
        self.timeframe = self.timeframe  # keep as string

    def _yf_interval(self) -> str:
        """
        Map our timeframe strings to yfinance interval strings.
        """
        tf = self.timeframe
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "1d": "1d",
        }

        if tf not in mapping:
            raise ValueError(f"Unsupported timeframe: {tf}")
        return mapping[tf]

    def _yf_period(self) -> str:
        """
        Choose a period long enough to get `limit` candles.
        We just use a fixed period depending on interval.
        """
        interval = self._yf_interval()
        # simple defaults – can tune later
        if interval in ("1m", "5m", "15m", "30m", "60m"):
            return "7d"   # intraday data, last 7 days
        else:
            return "60d"  # daily, last 60 days

    def fetch_latest(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch the most recent `limit` candles for this symbol.

        Returns a DataFrame with index 'ts' and columns:
        open, high, low, close, volume
        """
        interval = self._yf_interval()
        period = self._yf_period()

        data = yf.download(
            self.symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if data.empty:
            raise RuntimeError(f"No data returned for {self.symbol} at {interval}")

        # yfinance columns: Open, High, Low, Close, Adj Close, Volume
        df = data.tail(limit).copy()
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]]

        # ensure numeric types
        df = df.astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }
        )

        df.index.name = "ts"
        return df

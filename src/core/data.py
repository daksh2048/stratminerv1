from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import hashlib
import pickle
import time
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

# Load .env file at module import
load_dotenv()

Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "1d"]


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV data and drop invalid bars.
    Checks for:
    - High < Low
    - Close outside High/Low range
    - Open outside High/Low range
    - Negative/zero prices
    - Negative volume
    """
    if df is None or df.empty:
        return df

    invalid = (
        (df["high"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0) |
        (df["open"] <= 0) |
        (df["volume"] < 0)
    )

    if invalid.sum() > 0:
        print(f"⚠️  Data validation: Dropped {invalid.sum()} invalid OHLCV bars")
        df = df[~invalid].copy()

    return df


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

    # Remove duplicate timestamps if any
    if out.index.duplicated().any():
        n_dupes = out.index.duplicated().sum()
        print(f"⚠️  Removed {n_dupes} duplicate timestamps (kept last)")
        out = out[~out.index.duplicated(keep='last')]

    out.index.name = "ts"
    return out


@dataclass
class CandleFeed:
    exchange: str
    symbol: str
    timeframe: Timeframe = "5m"
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None

    def __post_init__(self):
        """Load Alpaca keys from .env file if not provided"""
        if self.exchange.lower() == "alpaca":
            if not self.alpaca_api_key:
                self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
            if not self.alpaca_secret_key:
                self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
            
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                raise ValueError(
                    "Alpaca exchange requires API credentials. "
                    "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
                )

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

    def _alpaca_timeframe(self) -> str:
        """Convert our timeframe to Alpaca format"""
        mapping = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "1d": "1Day",
        }
        if self.timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe for Alpaca: {self.timeframe}")
        return mapping[self.timeframe]

    def _fetch_alpaca(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data from Alpaca
        
        Note: Historical data always uses https://data.alpaca.markets
        The ALPACA_BASE_URL in your .env (paper-api.alpaca.markets) is for trading, not data.
        """
        url = f"https://data.alpaca.markets/v2/stocks/{self.symbol}/bars"
        
        headers = {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
        }
        
        params = {
            "timeframe": self._alpaca_timeframe(),
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z",
            "limit": 10000,  # Alpaca max per request
            "adjustment": "all",  # Include splits/dividends
        }
        
        all_bars = []
        next_page_token = None
        
        # Alpaca paginates results
        while True:
            if next_page_token:
                params["page_token"] = next_page_token
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Alpaca API request failed: {e}")
            
            if "bars" not in data or not data["bars"]:
                break
            
            all_bars.extend(data["bars"])
            
            # Check for next page
            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break
        
        if not all_bars:
            raise RuntimeError(
                f"No data returned from Alpaca for {self.symbol} "
                f"timeframe={self._alpaca_timeframe()} from {start_date} to {end_date}"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(all_bars)
        
        # Alpaca returns: t (timestamp), o, h, l, c, v, n (trade_count), vw (vwap)
        df = df.rename(columns={
            "t": "ts",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })
        
        # Convert timestamp to datetime and set as index
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        
        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].copy()
        
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()
        
        # Remove duplicates if any
        if df.index.duplicated().any():
            n_dupes = df.index.duplicated().sum()
            print(f"⚠️  Removed {n_dupes} duplicate timestamps from Alpaca (kept last)")
            df = df[~df.index.duplicated(keep='last')]
        
        df.index.name = "ts"
        return df

    def _calculate_date_range(self, period: str) -> tuple[datetime, datetime]:
        """Convert period string to start/end dates"""
        end = datetime.now()
        
        if period.endswith("d"):
            days = int(period[:-1])
            start = end - timedelta(days=days)
        elif period.endswith("mo"):
            months = int(period[:-2])
            start = end - timedelta(days=months * 30)  # Approximate
        elif period.endswith("y"):
            years = int(period[:-1])
            start = end - timedelta(days=years * 365)
        else:
            raise ValueError(f"Unsupported period format: {period}. Use format like '60d', '6mo', '5y'")
        
        return start, end

    def fetch(self, period: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical data from configured exchange.
        
        Args:
            period: Time period (e.g., "60d", "6mo", "5y")
            limit: Optional limit on number of bars to return (tail)
        
        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        # Route to appropriate data source
        if self.exchange.lower() == "yahoo":
            return self._fetch_yahoo(period, limit)
        elif self.exchange.lower() == "alpaca":
            return self._fetch_alpaca_with_cache(period, limit)
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")

    def _fetch_yahoo(self, period: Optional[str], limit: Optional[int]) -> pd.DataFrame:
        """Original Yahoo Finance fetch logic"""
        interval = self._yf_interval()

        # defaults
        if period is None:
            if interval == "1m":
                period = "7d"
            elif interval in ("5m", "15m", "30m", "60m"):
                period = "60d"
            else:
                period = "5y"

        # Cache setup
        cache_dir = Path("data_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_key = hashlib.md5(f"yahoo_{self.symbol}_{interval}_{period}".encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.pkl"

        # Check cache (24hr expiry)
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 86400:  # 24 hours
                try:
                    with open(cache_file, "rb") as f:
                        df = pickle.load(f)
                        if limit is not None:
                            df = df.tail(int(limit)).copy()
                        return df
                except Exception as e:
                    print(f"⚠️  Cache read failed for {self.symbol}, re-downloading: {e}")

        # Download
        try:
            data = yf.download(
                self.symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download data for {self.symbol}: {e}")

        if data is None or data.empty:
            raise RuntimeError(f"No data returned for {self.symbol} interval={interval} period={period}")

        df = _normalize_yf_columns(data)
        df = _validate_ohlcv(df)

        # Log what we got
        if not df.empty:
            print(f"✓ Fetched {self.symbol} {interval}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"⚠️  No valid data for {self.symbol} {interval} after validation")

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"⚠️  Cache write failed for {self.symbol}: {e}")

        if limit is not None:
            df = df.tail(int(limit)).copy()

        return df

    def _fetch_alpaca_with_cache(self, period: Optional[str], limit: Optional[int]) -> pd.DataFrame:
        """Fetch from Alpaca with caching"""
        # Default period for Alpaca
        if period is None:
            if self.timeframe in ("1m", "5m", "15m", "30m", "1h"):
                period = "60d"
            else:
                period = "5y"
        
        start_date, end_date = self._calculate_date_range(period)
        
        # Cache setup
        cache_dir = Path("data_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_key = hashlib.md5(
            f"alpaca_{self.symbol}_{self.timeframe}_{period}".encode()
        ).hexdigest()
        cache_file = cache_dir / f"{cache_key}.pkl"

        # Check cache (24hr expiry)
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 86400:  # 24 hours
                try:
                    with open(cache_file, "rb") as f:
                        df = pickle.load(f)
                        if limit is not None:
                            df = df.tail(int(limit)).copy()
                        return df
                except Exception as e:
                    print(f"⚠️  Cache read failed for {self.symbol}, re-downloading: {e}")

        # Fetch from Alpaca
        df = self._fetch_alpaca(start_date, end_date)
        df = _validate_ohlcv(df)

        # Log what we got
        if not df.empty:
            print(f"✓ Fetched {self.symbol} {self.timeframe} from Alpaca: "
                  f"{len(df)} bars from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"⚠️  No valid data for {self.symbol} {self.timeframe} from Alpaca after validation")

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"⚠️  Cache write failed for {self.symbol}: {e}")

        if limit is not None:
            df = df.tail(int(limit)).copy()

        return df

    def fetch_latest(self, limit: int = 500) -> pd.DataFrame:
        return self.fetch(period=None, limit=limit)
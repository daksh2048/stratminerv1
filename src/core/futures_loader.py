"""
Futures data loader for ES and other futures contracts
Loads from pre-parsed parquet files, slices to requested period
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def _parse_period(period: str) -> Optional[pd.DateOffset]:
    """Parse '10y', '3y', '18m', '90d' into a DateOffset. Returns None = load all."""
    if not period:
        return None
    s = str(period).strip().lower()
    try:
        if s.endswith("y"):
            return pd.DateOffset(years=int(s[:-1]))
        if s.endswith("m"):
            return pd.DateOffset(months=int(s[:-1]))
        if s.endswith("d"):
            return pd.DateOffset(days=int(s[:-1]))
    except (ValueError, TypeError):
        pass
    return None


def load_futures_data(
    symbol: str,
    timeframe: str,
    period: str = None,
    data_dir: str = "data"
) -> pd.DataFrame:
    """
    Load futures data from parquet files.

    Args:
        symbol:    'ES', 'NQ', 'USDJPY', etc.
        timeframe: '5min', '15min', '1h', etc.
        period:    How much history to keep: '10y', '3y', '18m', '90d'.
                   None = load entire file.
        data_dir:  Directory containing parquet files.

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    filepath = Path(data_dir) / f"{symbol}_{timeframe}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"Futures data file not found: {filepath}\n\n"
            f"For ES futures, run first:\n"
            f"  python prep_es_data.py\n\n"
            f"For forex (USDJPY etc.), run:\n"
            f"  python parse_txt_data.py\n"
            f"{'='*80}\n"
        )

    df = pd.read_parquet(filepath)
    total_bars = len(df)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index is not DatetimeIndex in {filepath}")

    df = df.sort_index()

    # Slice to requested period
    offset = _parse_period(period)
    if offset is not None:
        cutoff = df.index[-1] - offset
        df = df[df.index >= cutoff]
        print(f"  [{symbol} {timeframe}] period='{period}': "
              f"{len(df):,} bars kept of {total_bars:,} "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    else:
        print(f"  [{symbol} {timeframe}] loaded {total_bars:,} bars "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    # Validate required columns
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")

    if "volume" not in df.columns:
        print(f"  WARNING: no volume for {symbol} — using dummy 1s")
        df["volume"] = 1

    return df


def get_available_futures(data_dir: str = "data") -> list:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    available = []
    for fp in data_path.glob("*.parquet"):
        parts = fp.stem.split("_")
        if len(parts) >= 2:
            available.append((parts[0], "_".join(parts[1:])))
    return sorted(available)


if __name__ == "__main__":
    import sys
    available = get_available_futures()
    if not available:
        print("No parquet files found in data/. Run prep_es_data.py first.")
        sys.exit(1)
    print("Available data:")
    for sym, tf in available:
        print(f"  {sym} {tf}")
    df = load_futures_data("ES", "15min", period="10y")
    print(df.tail())
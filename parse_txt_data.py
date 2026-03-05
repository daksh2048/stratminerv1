"""
Parse .txt format data files into format compatible with backtest engine

Handles both:
- Forex data (USD/JPY, GBP/USD, AUD/USD, AUD/JPY)
- ES futures data

Input format:
Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume
2008/5/4, 22:00:00, 1413.50, 1414.00, 1412.75, 1413.75, 7, 0, 0, 0

Output format:
DataFrame with datetime index and OHLCV columns for backtester
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def parse_txt_to_df(
    filepath: str,
    resample_to: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse .txt data file to DataFrame
    
    Args:
        filepath: Path to .txt file
        resample_to: Optional resample (e.g., '15min', '1h')
    
    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    # Read CSV
    df = pd.read_csv(
        filepath,
        skipinitialspace=True,  # Handle spaces after commas
    )
    
    # Combine Date and Time into datetime
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%Y/%m/%d %H:%M:%S'
    )
    
    # Rename columns to standard OHLCV
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Last': 'close',  # "Last" is the close price
        'Volume': 'volume'
    })
    
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN
    df = df.dropna()
    
    # Resample if requested
    if resample_to:
        df = resample_bars(df, resample_to)
    
    return df


def resample_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample bars to different timeframe
    
    Args:
        df: DataFrame with OHLCV
        timeframe: '5min', '15min', '1h', etc.
    
    Returns:
        Resampled DataFrame
    """
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """Save DataFrame to parquet for faster loading"""
    df.to_parquet(output_path)
    print(f"Saved {len(df)} bars to {output_path}")


def load_from_parquet(filepath: str) -> pd.DataFrame:
    """Load DataFrame from parquet"""
    return pd.read_parquet(filepath)


# Example usage
if __name__ == "__main__":
    import os

    os.makedirs('data', exist_ok=True)

    # ── USDJPY forex (15min only) ─────────────────────────────────────────────
    usdjpy_path = r"C:\SierraChart\Data\USDJPY1mhis.scid_BarData.txt"
    if os.path.exists(usdjpy_path):
        usdjpy_df = parse_txt_to_df(usdjpy_path, resample_to='15min')
        print(f"USDJPY: {len(usdjpy_df)} bars  {usdjpy_df.index[0]} → {usdjpy_df.index[-1]}")
        save_to_parquet(usdjpy_df, "data/USDJPY_15min.parquet")
    else:
        print(f"Skipping USDJPY — file not found: {usdjpy_path}")

    # ── ES futures — generate ALL timeframes so strategies never crash ────────
    es_path = r"C:\SierraChart\Data\ESZ25-CME1m.scid_BarData.txt"
    if os.path.exists(es_path):
        es_1m = parse_txt_to_df(es_path)
        print(f"ES raw 1min: {len(es_1m)} bars  {es_1m.index[0]} → {es_1m.index[-1]}")

        # Slice to last 10 years
        cutoff = es_1m.index[-1] - pd.DateOffset(years=10)
        es_1m  = es_1m[es_1m.index >= cutoff]
        print(f"ES after 10y slice: {len(es_1m)} bars  ({es_1m.index[0].date()} → {es_1m.index[-1].date()})")

        # Generate every timeframe the strategies might request
        for label, rule in [("5min", "5min"), ("15min", "15min"),
                             ("30min", "30min"), ("1h", "1h")]:
            df_tf = resample_bars(es_1m, rule)
            save_to_parquet(df_tf, f"data/ES_{label}.parquet")

        # Also save with short-name aliases (e.g. "30m", "1h") that some
        # strategy configs may request via tf param
        for src_label, alias in [("30min", "30m"), ("1h", "60m")]:
            import shutil
            src = f"data/ES_{src_label}.parquet"
            dst = f"data/ES_{alias}.parquet"
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                print(f"  Aliased {src} → {dst}")
    else:
        print(f"Skipping ES — file not found: {es_path}")
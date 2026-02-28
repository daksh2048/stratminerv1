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
    # Parse ES futures data
    es_path = "C:\SierraChart\Data\ESZ25-CME1m.scid_BarData.txt"  # User should replace
    es_df = parse_txt_to_df(es_path, resample_to='15min')
    
    print(f"ES Data: {len(es_df)} bars")
    print(f"Date range: {es_df.index[0]} to {es_df.index[-1]}")
    print(es_df.head())
    
    # Save to parquet for faster future loads
    save_to_parquet(es_df, "ES_15min.parquet")
    
    # Parse forex data
    usdjpy_path = "C:\SierraChart\Data\USDJPY.scid_BarData.txt"  # User should replace
    usdjpy_df = parse_txt_to_df(usdjpy_path, resample_to='15min')
    
    print(f"\nUSD/JPY Data: {len(usdjpy_df)} bars")
    print(f"Date range: {usdjpy_df.index[0]} to {usdjpy_df.index[-1]}")
    print(usdjpy_df.head())
    
    save_to_parquet(usdjpy_df, "USDJPY_15min.parquet")
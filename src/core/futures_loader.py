"""
Futures data loader for ES and other futures contracts
Loads from pre-parsed parquet files
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_futures_data(
    symbol: str,
    timeframe: str,
    period: str = None,  # Ignored, loads full file
    data_dir: str = "data"
) -> pd.DataFrame:
    """
    Load futures data from parquet files
    
    Args:
        symbol: 'ES', 'NQ', 'YM', etc.
        timeframe: '5min', '15min', '1h', etc.
        period: Ignored (loads entire file)
        data_dir: Directory containing parquet files
    
    Returns:
        DataFrame with OHLCV data and datetime index
    
    Raises:
        FileNotFoundError: If parquet file doesn't exist
    """
    filepath = Path(data_dir) / f"{symbol}_{timeframe}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"Futures data file not found: {filepath}\n"
            f"\n"
            f"You need to parse your .txt file first:\n"
            f"\n"
            f"  1. Run: python parse_txt_data.py\n"
            f"  2. Or manually:\n"
            f"\n"
            f"     from parse_txt_data import parse_txt_to_df, save_to_parquet\n"
            f"     df = parse_txt_to_df('path/to/{symbol}.txt', resample_to='{timeframe}')\n"
            f"     save_to_parquet(df, '{filepath}')\n"
            f"\n"
            f"{'='*80}\n"
        )
    
    # Load parquet
    df = pd.read_parquet(filepath)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")
    
    # Add volume if missing (some futures feeds don't include volume)
    if 'volume' not in df.columns:
        print(f"WARNING: No volume data for {symbol}, using dummy volume")
        df['volume'] = 1
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index is not DatetimeIndex in {filepath}")
    
    print(f"Loaded {symbol} {timeframe}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    return df


def get_available_futures(data_dir: str = "data") -> list:
    """
    Get list of available futures symbols
    
    Returns:
        List of (symbol, timeframe) tuples
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    parquet_files = list(data_path.glob("*.parquet"))
    
    available = []
    for filepath in parquet_files:
        # Parse filename: SYMBOL_TIMEFRAME.parquet
        name = filepath.stem  # Remove .parquet
        parts = name.split('_')
        
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = '_'.join(parts[1:])
            available.append((symbol, timeframe))
    
    return sorted(available)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Show available data
    available = get_available_futures()
    
    if not available:
        print("No futures data found in data/ directory")
        print("\nRun parse_txt_data.py first to convert .txt files to parquet")
        sys.exit(1)
    
    print("Available futures data:")
    for symbol, timeframe in available:
        print(f"  {symbol} - {timeframe}")
    
    # Try loading ES 5min
    try:
        df = load_futures_data('ES', '5min')
        print(f"\nES 5min data:")
        print(f"  Bars: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\nFirst 5 bars:")
        print(df.head())
    except FileNotFoundError as e:
        print(f"\n{e}")
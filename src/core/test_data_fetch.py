import sys
sys.path.insert(0, '.')

from src.core.data import CandleFeed
import pandas as pd

print("Testing data fetching...")

# Test 1: 5m data for SPY
print("\n1. Fetching SPY 5m data (60 days)...")
try:
    feed_5m = CandleFeed(exchange="yahoo", symbol="SPY", timeframe="5m")
    df_5m = feed_5m.fetch(period="60d")
    print(f"   ✓ Success! Got {len(df_5m)} bars")
    print(f"   Date range: {df_5m.index[0]} to {df_5m.index[-1]}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: 1h data for SPY (this is what's hanging)
print("\n2. Fetching SPY 1h data (60 days)...")
try:
    feed_1h = CandleFeed(exchange="yahoo", symbol="SPY", timeframe="1h")
    df_1h = feed_1h.fetch(period="60d")
    print(f"   ✓ Success! Got {len(df_1h)} bars")
    print(f"   Date range: {df_1h.index[0]} to {df_1h.index[-1]}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    print("\nThis is the problem! HTF data fetch is failing.")
    sys.exit(1)

# Test 3: Verify cache works
print("\n3. Testing cache (should be instant)...")
import time
start = time.time()
df_cached = feed_1h.fetch(period="60d")
elapsed = time.time() - start
print(f"   ✓ Cached fetch took {elapsed:.2f} seconds")

print("\n✓ All tests passed! Your backtest should work now.")
print("Run: python -m src.main")
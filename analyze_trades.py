import pandas as pd

df = pd.read_csv("trades.csv")

# Compute risk and R multiple for each trade
risk = (df["entry"] - df["stop"]).abs() * df["size"]
df["R"] = df["pnl"] / risk.replace(0, float("nan"))

# Extract timeframe from strategy name (assumes *_5m / *_15m / *_30m)
def extract_tf(s: str) -> str:
    parts = s.split("_")
    return parts[-1] if parts[-1] in ("5m", "15m", "30m", "1h") else "unknown"

df["tf"] = df["strategy"].apply(extract_tf)

# Extract hour of day from open_time
df["open_time"] = pd.to_datetime(df["open_time"])
df["hour"] = df["open_time"].dt.hour

# 1) Overall per-strategy expectancy
print("\n=== Per-strategy R stats ===")
print(df.groupby("strategy")["R"].agg(["count", "mean", "sum"]))

# 2) Per (strategy, tf) stats
print("\n=== Per (strategy, tf) R stats ===")
print(df.groupby(["strategy", "tf"])["R"].agg(["count", "mean", "sum"]))

# 3) Conditional on time of day (example buckets)
df["session"] = pd.cut(
    df["hour"],
    bins=[9, 11, 13, 16, 20],
    labels=["open", "midday", "pm", "late"],
    right=False,
)

print("\n=== Per (strategy, session) R stats ===")
print(df.groupby(["strategy", "session"])["R"].agg(["count", "mean", "sum"]))
import pandas as pd

df = pd.read_csv("trades.csv")

# Compute risk and R multiple for each trade
risk = (df["entry"] - df["stop"]).abs() * df["size"]
df["R"] = df["pnl"] / risk.replace(0, float("nan"))

# Extract timeframe from strategy name (assumes *_5m / *_15m / *_30m)
def extract_tf(s: str) -> str:
    parts = s.split("_")
    return parts[-1] if parts[-1] in ("5m", "15m", "30m", "1h") else "unknown"

df["tf"] = df["strategy"].apply(extract_tf)

# Extract hour of day from open_time
df["open_time"] = pd.to_datetime(df["open_time"])
df["hour"] = df["open_time"].dt.hour

# 1) Overall per-strategy expectancy
print("\n=== Per-strategy R stats ===")
print(df.groupby("strategy")["R"].agg(["count", "mean", "sum"]))

# 2) Per (strategy, tf) stats
print("\n=== Per (strategy, tf) R stats ===")
print(df.groupby(["strategy", "tf"])["R"].agg(["count", "mean", "sum"]))

# 3) Conditional on time of day (example buckets)
df["session"] = pd.cut(
    df["hour"],
    bins=[9, 11, 13, 16, 20],
    labels=["open", "midday", "pm", "late"],
    right=False,
)

print("\n=== Per (strategy, session) R stats ===")
print(df.groupby(["strategy", "session"])["R"].agg(["count", "mean", "sum"]))

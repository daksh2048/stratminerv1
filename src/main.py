import os
import yaml
import pandas as pd

from src.core.data import CandleFeed
from src.core.execution import PaperBroker
from src.strategies.rsi_extremes_reversal import RSIExtremesReversal


# -----------------------------
# CSV helpers
# -----------------------------
TRADES_HEADER = [
    "symbol",
    "side",
    "size",
    "entry",
    "stop",
    "take",
    "open_time",
    "close_time",
    "exit_price",
    "pnl",
    "R",
    "result",
    "strategy",
    "status",
    "balance",
]


def ensure_csv_header(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write(",".join(TRADES_HEADER) + "\n")


def to_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def to_market_tz(ts: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    """
    Keep consistent with the rest of your project:
    - If df index is tz-naive, your CandleFeed often represents UTC -> localize to UTC then convert.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(market_tz)


# -----------------------------
# One isolated backtest run
# -----------------------------
def backtest_rsi_one(sym: str, tf: str, cfg: dict) -> dict:
    eng = cfg["engine"]
    log_cfg = cfg.get("logging", {}) or {}

    strat_cfg = (cfg.get("strategies") or {}).get("rsi_rev", {}) or {}

    rsi = RSIExtremesReversal(name="rsi_rev", **strat_cfg)

    # ---- Data ----
    limit = int(eng.get("limit", 2000))
    feed = CandleFeed(exchange=eng.get("exchange", "yahoo"), symbol=sym, timeframe=tf)
    df = feed.fetch_latest(limit=limit).sort_index()

    # ---- Fresh broker per run ----
    out_dir = log_cfg.get("out_dir", "backtests")
    trades_path = os.path.join(out_dir, f"trades_rsiRev_{sym}_{tf}.csv")

    # start clean each run (prevents appending onto old output)
    if os.path.exists(trades_path):
        os.remove(trades_path)
    ensure_csv_header(trades_path)

    broker = PaperBroker(
        starting_balance=float(eng.get("paper_balance", 10000.0)),
        risk_per_trade=float(eng.get("risk_per_trade", 0.005)),
        slippage_bps=float(eng.get("slippage_bps", 0.0)),
        trades_csv=trades_path,
        max_loss_pct=float(eng.get("max_loss_pct", 0.03)),
    )

    # ---- Warmup ----
    # RSI needs enough bars for RSI/ATR/EMA if used
    rsi_period = int(strat_cfg.get("rsi_period", 14))
    atr_period = int(strat_cfg.get("atr_period", 14))
    ema_period = int(strat_cfg.get("ema_period", 50))
    warmup = max(rsi_period, atr_period, ema_period) + 30

    if len(df) <= warmup:
        return {
            "symbol": sym,
            "tf": tf,
            "trades_csv": trades_path,
            "final_balance": broker.balance,
            "closed": 0,
            "skipped": f"not enough candles (have={len(df)}, need>{warmup})",
        }

    # ---- EOD flatten settings ----
    market_tz = str(strat_cfg.get("market_tz", "America/New_York"))
    rth_close = str(strat_cfg.get("rth_close", "16:00"))
    close_h, close_m = map(int, rth_close.split(":"))

    current_day = None
    session_close = None
    prev_now = None
    prev_candle = None

    # ---- Event loop: candle-by-candle ----
    for i in range(warmup, len(df) + 1):
        candle = df.iloc[i - 1]
        now = candle.name

        now_m = to_market_tz(now, market_tz)
        day_m = now_m.normalize()

        # BULLETPROOF: flatten on day rollover using last candle of previous day we actually have
        if current_day is not None and day_m != current_day and prev_now is not None and prev_candle is not None:
            broker.force_close_all(
                now=prev_now,
                market_close_price=to_float(prev_candle["close"]),
                symbol=sym,
                tf=tf,
                status="eod_flat",
            )

        if current_day is None or day_m != current_day:
            current_day = day_m
            session_close = current_day + pd.Timedelta(hours=close_h, minutes=close_m)

        # 1) update/close positions on THIS candle
        broker.update_with_candle(candle, now, sym, tf)

        # 2) strategy sees candles up to now (no peeking)
        context = df.iloc[:i]
        order = rsi.on_candles(context, sym)

        if order is None or order.side not in ("buy", "sell"):
            prev_now = now
            prev_candle = candle
            continue

        order.meta = order.meta or {}
        order.meta["tf"] = tf

        broker.open_from_order(
            order=order,
            strategy=f"rsi_rev_{tf}",
            now=now,
            market_close_price=to_float(candle["close"]),
        )

        # optional: flatten if a >=16:00 bar exists
        if session_close is not None and now_m >= session_close:
            broker.force_close_all(
                now=now,
                market_close_price=to_float(candle["close"]),
                symbol=sym,
                tf=tf,
                status="eod_flat",
            )

        prev_now = now
        prev_candle = candle

    # ---- Close any leftovers at data end ----
    last = df.iloc[-1]
    broker.force_close_all(
        now=last.name,
        market_close_price=to_float(last["close"]),
        symbol=sym,
        tf=tf,
        status="data_end",
    )

    return {
        "symbol": sym,
        "tf": tf,
        "trades_csv": trades_path,
        "final_balance": broker.balance,
        "closed": len(broker.closed_positions),
        "skipped": None,
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eng = cfg["engine"]

    symbols = eng.get("symbols", ["SPY", "NVDA", "GLD", "QQQ"])
    timeframes = eng.get("timeframes", ["5m"])

    # RSI reversal is an intraday setup; keep it on intraday TFs
    allowed = {"1m", "5m", "15m", "30m"}
    timeframes = [tf for tf in timeframes if tf in allowed]
    if not timeframes:
        timeframes = ["5m"]

    results = []
    print("\n=== RSI isolated backtests (fresh broker per run) ===")
    for tf in timeframes:
        for sym in symbols:
            r = backtest_rsi_one(sym, tf, cfg)
            results.append(r)

            if r["skipped"]:
                print(f"[SKIP] {sym} {tf} -> {r['skipped']}")
            else:
                print(
                    f"[DONE] {sym} {tf} | closed={r['closed']} | final_balance={r['final_balance']:.2f} | csv={r['trades_csv']}"
                )

    print("\n=== Summary ===")
    for r in results:
        status = "SKIP" if r["skipped"] else "OK"
        print(
            f"{status:4} | {r['symbol']:4} {r['tf']:3} | closed={r['closed']:3} | final={r['final_balance']:.2f} | {r['trades_csv']}"
        )

import os
import math
import yaml
import pandas as pd

from src.core.data import CandleFeed
from src.core.execution import PaperBroker
from src.strategies.opening_range_breakout import OpeningRangeBreakout


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
    """
    PaperBroker appends rows with no header. This makes analysis painful.
    We write the header once if file is new/empty.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write(",".join(TRADES_HEADER) + "\n")


def to_float(x) -> float:
    """
    Avoid pandas FutureWarning where float() gets called on a 1-element Series.
    """
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


# -----------------------------
# TF helpers
# -----------------------------
_TF_TO_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440}


def tf_minutes(tf: str) -> int:
    if tf not in _TF_TO_MIN:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return _TF_TO_MIN[tf]


def opening_minutes_to_or_bars(opening_range_minutes: int, ltf: str) -> int:
    mins = tf_minutes(ltf)
    return max(1, int(math.ceil(opening_range_minutes / mins)))


# -----------------------------
# One isolated backtest run
# -----------------------------
def backtest_orb_one(sym: str, tf: str, cfg: dict) -> dict:
    eng = cfg["engine"]
    log_cfg = cfg.get("logging", {}) or {}

    # ---- ORB params ----
    strat_cfg = (cfg.get("strategies") or {}).get("orb", {}) or {}

    opening_range_minutes = int(strat_cfg.get("opening_range_minutes", 30))
    risk_reward = float(strat_cfg.get("risk_reward", 2.0))
    once_per_day = bool(strat_cfg.get("once_per_day", True))

    # If user explicitly sets or_bars, it wins. Otherwise derive from opening_range_minutes + timeframe.
    or_bars = int(strat_cfg.get("or_bars", opening_minutes_to_or_bars(opening_range_minutes, tf)))

    orb = OpeningRangeBreakout(
        name="orb",
        or_bars=or_bars,
        risk_reward=risk_reward,
        once_per_day=once_per_day,
    )

    # ---- Data ----
    limit = int(eng.get("limit", 2000))
    feed = CandleFeed(exchange=eng.get("exchange", "yahoo"), symbol=sym, timeframe=tf)
    df = feed.fetch_latest(limit=limit).sort_index()

    # ---- Fresh broker per run (THIS is the fix) ----
    out_dir = log_cfg.get("out_dir", "backtests")
    trades_path = os.path.join(out_dir, f"trades_orb_{sym}_{tf}.csv")
    ensure_csv_header(trades_path)

    broker = PaperBroker(
        starting_balance=float(eng.get("paper_balance", 10000.0)),
        risk_per_trade=float(eng.get("risk_per_trade", 0.005)),
        slippage_bps=float(eng.get("slippage_bps", 0.0)),
        trades_csv=trades_path,
        max_loss_pct=float(eng.get("max_loss_pct", 0.03)),
    )

    # ---- Warmup ----
    warmup = max(or_bars + 5, 20)
    if len(df) <= warmup:
        return {
            "symbol": sym,
            "tf": tf,
            "trades_csv": trades_path,
            "final_balance": broker.balance,
            "closed": 0,
            "skipped": f"not enough candles (have={len(df)}, need>{warmup})",
        }

    # ---- Event loop: candle-by-candle ----
    for i in range(warmup, len(df) + 1):
        candle = df.iloc[i - 1]
        now = candle.name

        # 1) update/close positions on THIS candle
        broker.update_with_candle(candle, now, sym, tf)

        # 2) strategy sees candles up to now (no peeking)
        context = df.iloc[:i]
        order = orb.on_candles(context, sym)

        # no signal
        if order is None or order.side not in ("buy", "sell"):
            continue

        # tag tf so broker updates the right positions
        order.meta = order.meta or {}
        order.meta["tf"] = tf

        # IMPORTANT: store strategy as orb_{tf} so analysis can group by tf
        broker.open_from_order(
            order=order,
            strategy=f"orb_{tf}",
            now=now,
            market_close_price=to_float(candle["close"]),
        )

    # ---- Close any leftovers at data end (so no "open positions remaining") ----
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

    symbols = eng.get("symbols", ["SPY", "NVDA", "GLD"])
    timeframes = eng.get("timeframes", ["5m"])

    # Guardrail: ORB is an intraday opening session concept.
    # Running it on 1h/1d turns it into a different strategy.
    allowed = {"1m", "5m", "15m", "30m"}
    timeframes = [tf for tf in timeframes if tf in allowed]
    if not timeframes:
        # default fallback
        timeframes = ["5m"]

    results = []
    print("\n=== ORB isolated backtests (fresh broker per run) ===")
    for tf in timeframes:
        for sym in symbols:
            r = backtest_orb_one(sym, tf, cfg)
            results.append(r)

            if r["skipped"]:
                print(f"[SKIP] {sym} {tf} -> {r['skipped']}")
            else:
                print(
                    f"[DONE] {sym} {tf} | closed={r['closed']} | final_balance={r['final_balance']:.2f} | csv={r['trades_csv']}"
                )

    print("\n=== Summary ===")
    # quick summary table in stdout
    for r in results:
        status = "SKIP" if r["skipped"] else "OK"
        print(
            f"{status:4} | {r['symbol']:4} {r['tf']:3} | closed={r['closed']:3} | final={r['final_balance']:.2f} | {r['trades_csv']}"
        )

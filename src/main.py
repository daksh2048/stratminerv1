import os
import math
import yaml
import pandas as pd

from src.core.data import CandleFeed
from src.core.execution import PaperBroker

from src.strategies.opening_range_breakout import OpeningRangeBreakout
from src.strategies.vwap_mean_reversion import VWAPMeanReversion
from src.strategies.vwap_trend_pullback import VWAPTrendPullback


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


_TF_TO_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440}


def tf_minutes(tf: str) -> int:
    if tf not in _TF_TO_MIN:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return _TF_TO_MIN[tf]


def opening_minutes_to_or_bars(opening_range_minutes: int, ltf: str) -> int:
    mins = tf_minutes(ltf)
    return max(1, int(math.ceil(opening_range_minutes / mins)))


def make_strategy(name: str, cfg: dict, tf: str):
    strat_cfg = (cfg.get("strategies") or {}).get(name, {}) or {}

    if name == "orb":
        opening_range_minutes = int(strat_cfg.get("opening_range_minutes", 30))
        or_bars = int(strat_cfg.get("or_bars", opening_minutes_to_or_bars(opening_range_minutes, tf)))
        strat_cfg = dict(strat_cfg)
        strat_cfg["or_bars"] = or_bars
        return OpeningRangeBreakout(name="orb", **strat_cfg)

    if name == "vwap_mr":
        return VWAPMeanReversion(name="vwap_mr", **strat_cfg)

    if name == "vwap_tp":
        return VWAPTrendPullback(name="vwap_tp", **strat_cfg)

    raise ValueError(f"Unknown strategy: {name}")


def backtest_one(strategy_name: str, sym: str, tf: str, cfg: dict) -> dict:
    eng = cfg["engine"]
    log_cfg = cfg.get("logging", {}) or {}

    period = str(eng.get("period", "60d"))
    limit = eng.get("limit", None)  # optional tail, usually None for full period

    strat = make_strategy(strategy_name, cfg, tf)

    # ---- Data ----
    feed = CandleFeed(exchange=eng.get("exchange", "yahoo"), symbol=sym, timeframe=tf)
    df = feed.fetch(period=period, limit=limit).sort_index()

    # ---- Fresh broker per run ----
    out_dir = log_cfg.get("out_dir", "backtests")
    trades_path = os.path.join(out_dir, f"trades_{strategy_name}_{sym}_{tf}.csv")

    # overwrite output each run
    if os.path.exists(trades_path):
        os.remove(trades_path)
    ensure_csv_header(trades_path)

    starting_balance = float(eng.get("paper_balance", 10000.0))

    broker = PaperBroker(
        starting_balance=starting_balance,
        risk_per_trade=float(eng.get("risk_per_trade", 0.005)),
        slippage_bps=float(eng.get("slippage_bps", 0.0)),
        trades_csv=trades_path,
        max_loss_pct=float(eng.get("max_loss_pct", 0.03)),
    )

    warmup = int(eng.get("warmup_bars", 80))
    if len(df) <= warmup:
        return {
            "strategy": strategy_name,
            "symbol": sym,
            "tf": tf,
            "trades_csv": trades_path,
            "final_balance": broker.balance,
            "closed": 0,
            "skipped": f"not enough candles (have={len(df)}, need>{warmup})",
        }

    # ---- Event loop ----
    for i in range(warmup, len(df) + 1):
        candle = df.iloc[i - 1]
        now = candle.name

        broker.update_with_candle(candle, now, sym, tf)

        context = df.iloc[:i]
        order = strat.on_candles(context, sym)

        if order is None or order.side not in ("buy", "sell"):
            continue

        order.meta = order.meta or {}
        order.meta["tf"] = tf

        broker.open_from_order(
            order=order,
            strategy=f"{strategy_name}_{tf}",
            now=now,
            market_close_price=to_float(candle["close"]),
        )

    # close leftovers at end
    last = df.iloc[-1]
    broker.force_close_all(
        now=last.name,
        market_close_price=to_float(last["close"]),
        symbol=sym,
        tf=tf,
        status="data_end",
    )

    return {
        "strategy": strategy_name,
        "symbol": sym,
        "tf": tf,
        "trades_csv": trades_path,
        "final_balance": broker.balance,
        "closed": len(broker.closed_positions),
        "skipped": None,
    }


# -----------------------------
# Performance aggregation
# -----------------------------
def _read_trades_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=TRADES_HEADER)

    df = pd.read_csv(path)

    # Guard: empty file with just header
    if df.empty:
        return df

    # normalize
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    if "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    # only closed trades with a timestamp
    df = df.dropna(subset=["close_time"])
    return df


def compute_returns(results: list[dict], starting_balance: float, strategies: list[str], symbols: list[str]) -> None:
    """
    Prints:
      - Monthly return % per strategy across all symbols
      - Weekly return % per strategy across all symbols
      - Monthly/Weekly return % combined across all strategies+symbols
    """
    n_syms = max(1, len(symbols))
    n_strats = max(1, len(strategies))

    # collect pnl rows with strategy label
    rows = []
    for r in results:
        if r.get("skipped"):
            continue
        path = r["trades_csv"]
        tdf = _read_trades_csv(path)
        if tdf.empty:
            continue
        tdf = tdf.copy()
        tdf["strategy_key"] = r["strategy"]
        rows.append(tdf[["close_time", "pnl", "strategy_key"]])

    if not rows:
        print("\n=== Returns (no closed trades found) ===")
        return

    trades = pd.concat(rows, ignore_index=True)

    # Weekly bucket (Mon-Sun). You can change to "W-FRI" if you want “trading week”.
    trades["week"] = trades["close_time"].dt.to_period("W").astype(str)

    # Monthly totals per strategy
    pnl_by_strat = trades.groupby("strategy_key")["pnl"].sum().sort_index()

    # Weekly totals per strategy
    pnl_week_strat = trades.groupby(["strategy_key", "week"])["pnl"].sum().reset_index()

    # denominators
    denom_per_strat = starting_balance * n_syms
    denom_all = starting_balance * n_syms * n_strats

    print("\n=== Monthly Return % (per strategy across ALL symbols) ===")
    for s in strategies:
        pnl = float(pnl_by_strat.get(s, 0.0))
        ret_pct = (pnl / denom_per_strat) * 100.0
        print(f"{s:8} | pnl={pnl:10.2f} | denom={denom_per_strat:10.2f} | return={ret_pct:7.3f}%")

    total_pnl_all = float(trades["pnl"].sum())
    total_ret_all = (total_pnl_all / denom_all) * 100.0
    print(f"\nALL     | pnl={total_pnl_all:10.2f} | denom={denom_all:10.2f} | return={total_ret_all:7.3f}%")

    print("\n=== Weekly Return % (per strategy across ALL symbols) ===")
    # print in week order
    weeks = sorted(trades["week"].unique())

    for s in strategies:
        print(f"\n-- {s} --")
        sub = pnl_week_strat[pnl_week_strat["strategy_key"] == s].set_index("week")["pnl"]
        for w in weeks:
            pnl = float(sub.get(w, 0.0))
            ret_pct = (pnl / denom_per_strat) * 100.0
            print(f"{w} | pnl={pnl:10.2f} | return={ret_pct:7.3f}%")

    print("\n=== Weekly Return % (ALL strategies combined) ===")
    pnl_week_all = trades.groupby("week")["pnl"].sum()
    for w in weeks:
        pnl = float(pnl_week_all.get(w, 0.0))
        ret_pct = (pnl / denom_all) * 100.0
        print(f"{w} | pnl={pnl:10.2f} | return={ret_pct:7.3f}%")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eng = cfg["engine"]
    symbols = eng.get("symbols", ["SPY", "QQQ", "GLD"])
    timeframes = eng.get("timeframes", ["5m"])
    strategies_to_run = eng.get("strategies_to_run", ["orb", "vwap_mr", "vwap_tp"])

    results = []
    print("\n=== Phase-1 batch backtests (period-based) ===")
    for strat_name in strategies_to_run:
        print(f"\n--- Strategy: {strat_name} ---")
        for tf in timeframes:
            for sym in symbols:
                r = backtest_one(strat_name, sym, tf, cfg)
                results.append(r)

                if r["skipped"]:
                    print(f"[SKIP] {strat_name} | {sym} {tf} -> {r['skipped']}")
                else:
                    print(
                        f"[DONE] {strat_name} | {sym} {tf} | closed={r['closed']} | final={r['final_balance']:.2f} | {r['trades_csv']}"
                    )

    print("\n=== Summary ===")
    for r in results:
        status = "SKIP" if r["skipped"] else "OK"
        print(
            f"{status:4} | {r['strategy']:7} | {r['symbol']:5} {r['tf']:3} | closed={r['closed']:3} | final={r['final_balance']:.2f} | {r['trades_csv']}"
        )

    # ---- returns ----
    starting_balance = float(eng.get("paper_balance", 10000.0))
    compute_returns(results, starting_balance, strategies_to_run, symbols)

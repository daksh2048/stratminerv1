import os
<<<<<<< HEAD
import math
=======
>>>>>>> 8865de7 (Made multiple changes, modified vwap_mean_reversion with a better/more)
import yaml
import pandas as pd

from src.core.data import CandleFeed
from src.core.execution import PaperBroker
<<<<<<< HEAD
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
=======
from src.strategies.rsi_extremes_reversal import RSIExtremesReversal


TRADES_HEADER = [
    "symbol","side","size","entry","stop","take","open_time","close_time","exit_price",
    "pnl","R","result","strategy","status","balance",
>>>>>>> 8865de7 (Made multiple changes, modified vwap_mean_reversion with a better/more)
]


def ensure_csv_header(path: str) -> None:
<<<<<<< HEAD
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
=======
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
    yfinance intraday often returns tz-naive timestamps that already represent market time.
    Rule:
      naive -> localize to market_tz
      aware -> convert to market_tz
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(market_tz)
    return ts.tz_convert(market_tz)


def run_backtest_for(sym: str, tf: str, cfg: dict) -> dict:
    eng = cfg["engine"]
    log_cfg = cfg.get("logging", {}) or {}
    strat_cfg = ((cfg.get("strategies") or {}).get("rsi_rev", {}) or {})

    # 1) fetch data
    limit = int(eng.get("limit", 2000))
    feed = CandleFeed(exchange=eng.get("exchange", "yahoo"), symbol=sym, timeframe=tf)
    try:
        df = feed.fetch_latest(limit=limit).sort_index()
    except Exception as e:
        out_dir = log_cfg.get("out_dir", "backtests")
        trades_path = os.path.join(out_dir, f"trades_rsiRev_{sym}_{tf}.csv")
        return {"symbol": sym, "tf": tf, "trades_csv": trades_path, "final_balance": float(eng.get("paper_balance", 10000.0)), "closed": 0, "skipped": f"data fetch failed: {e}"}

    # 2) prepare output csv
    out_dir = log_cfg.get("out_dir", "backtests")
    trades_path = os.path.join(out_dir, f"trades_rsiRev_{sym}_{tf}.csv")
    if os.path.exists(trades_path):
        os.remove(trades_path)
    ensure_csv_header(trades_path)

    # 3) broker + strategy
    broker = PaperBroker(
        starting_balance=float(eng.get("paper_balance", 10000.0)),
        risk_per_trade=float(eng.get("risk_per_trade", 0.005)),
        slippage_bps=float(eng.get("slippage_bps", 0.0)),
        trades_csv=trades_path,
        max_loss_pct=float(eng.get("max_loss_pct", 0.03)),
    )
    strat = RSIExtremesReversal(name="rsi_rev", **strat_cfg)

    # 4) session config (only for flattening)
    market_tz = str(strat_cfg.get("market_tz", "America/New_York"))
    rth_close = str(strat_cfg.get("rth_close", "16:00"))
    close_h, close_m = map(int, rth_close.split(":"))

    # 5) warmup
    atr_period = int(strat_cfg.get("atr_period", 14))
    rsi_period = int(strat_cfg.get("rsi_period", 14))
    warmup = max(atr_period + 30, rsi_period + 10)

    if len(df) <= warmup:
        return {"symbol": sym, "tf": tf, "trades_csv": trades_path, "final_balance": broker.balance, "closed": 0, "skipped": "not enough candles"}

    current_day = None
    session_close = None
    prev_now = None
    prev_candle = None

>>>>>>> 8865de7 (Made multiple changes, modified vwap_mean_reversion with a better/more)
    for i in range(warmup, len(df) + 1):
        candle = df.iloc[i - 1]
        now = candle.name

<<<<<<< HEAD
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
=======
        now_m = to_market_tz(now, market_tz)
        day_m = now_m.normalize()

        # BULLETPROOF: flatten on day rollover using last candle of previous day
        if current_day is not None and day_m != current_day and prev_now is not None and prev_candle is not None:
            broker.force_close_all(
                now=prev_now,
                market_close_price=to_float(prev_candle["close"]),
                symbol=sym,
                tf=tf,
                status="eod_flat",
            )

        # initialize day
        if current_day is None or day_m != current_day:
            current_day = day_m
            session_close = current_day + pd.Timedelta(hours=close_h, minutes=close_m)

        # 1) update exits
        broker.update_with_candle(candle, now, sym, tf)

        # 2) strategy signal
        order = strat.on_candles(df.iloc[:i], sym)

        if order is not None and order.side in ("buy", "sell"):
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

    # final safety close
>>>>>>> 8865de7 (Made multiple changes, modified vwap_mean_reversion with a better/more)
    last = df.iloc[-1]
    broker.force_close_all(
        now=last.name,
        market_close_price=to_float(last["close"]),
        symbol=sym,
        tf=tf,
        status="data_end",
    )

<<<<<<< HEAD
    return {
        "symbol": sym,
        "tf": tf,
        "trades_csv": trades_path,
        "final_balance": broker.balance,
        "closed": len(broker.closed_positions),
        "skipped": None,
    }
=======
    return {"symbol": sym, "tf": tf, "trades_csv": trades_path, "final_balance": broker.balance, "closed": len(broker.closed_positions), "skipped": None}
>>>>>>> 8865de7 (Made multiple changes, modified vwap_mean_reversion with a better/more)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eng = cfg["engine"]
<<<<<<< HEAD

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
=======
    symbols = eng.get("symbols", ["SPY", "NVDA", "GLD", "QQQ"])
    timeframes = eng.get("timeframes", ["5m"])

    results = []
    for sym in symbols:
        for tf in timeframes:
            results.append(run_backtest_for(sym, tf, cfg))

    print("\n=== Summary ===")
    for r in results:
        if r.get("skipped"):
            print(f"SKIP | {r['symbol']:<4} {r['tf']:<3} | {r['skipped']}")
        else:
            print(f"OK   | {r['symbol']:<4} {r['tf']:<3} | closed={r['closed']:>3} | final={r['final_balance']:.2f} | {r['trades_csv']}")
>>>>>>> 8865de7 (Made multiple changes, modified vwap_mean_reversion with a better/more)

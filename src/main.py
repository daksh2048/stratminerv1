import os
import math
import yaml
import pandas as pd

from src.core.data import CandleFeed
from src.core.execution import PaperBroker

from src.strategies.opening_range_breakout import OpeningRangeBreakout
from src.strategies.vwap_rejection import VWAPRejection
from src.strategies.vwap_momentum import VWAPMomentum
from src.strategies.swing_bos import SwingBOS

from src.strategies.gap_fill import GapFill
from src.strategies.bollinger_squeeze import BollingerSqueeze
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.rsi_mean_reversion import RSIMeanReversion
from src.strategies.sr_breakout import SupportResistanceBreakout


TRADES_HEADER = [
    "symbol", "side", "size", "entry", "stop", "take",
    "open_time", "close_time", "exit_price", "pnl", "R",
    "result", "strategy", "status", "balance",
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
    return max(1, int(math.ceil(opening_range_minutes / tf_minutes(ltf))))


def make_strategy(name: str, cfg: dict, tf: str):
    strat_cfg = (cfg.get("strategies") or {}).get(name, {}) or {}

    if name == "orb":
        opening_range_minutes = int(strat_cfg.get("opening_range_minutes", 30))
        or_bars = int(strat_cfg.get("or_bars", opening_minutes_to_or_bars(opening_range_minutes, tf)))
        strat_cfg = dict(strat_cfg)
        strat_cfg["or_bars"] = or_bars
        return OpeningRangeBreakout(name="orb", **strat_cfg)

    if name == "vwap_rejection":
        return VWAPRejection(name="vwap_rejection", **strat_cfg)

    if name == "vwap_momentum":
        return VWAPMomentum(name="vwap_momentum", **strat_cfg)

    if name == "swing_bos":
        return SwingBOS(name="swing_bos", **strat_cfg)

    if name == "gap_fill":
        return GapFill(name="gap_fill", **strat_cfg)
    
    if name == "bollinger_squeeze":
        return BollingerSqueeze(name="bollinger_squeeze", **strat_cfg)
    
    if name == "ma_crossover":
        return MovingAverageCrossover(name="ma_crossover", **strat_cfg)
    
    if name == "rsi_mean_reversion":
        return RSIMeanReversion(name="rsi_mean_reversion", **strat_cfg)
    
    if name == "sr_breakout":
        return SupportResistanceBreakout(name="sr_breakout", **strat_cfg)

    raise ValueError(f"Unknown strategy: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest runner
# ─────────────────────────────────────────────────────────────────────────────
def backtest_one(strategy_name: str, sym: str, tf: str, cfg: dict) -> dict:
    eng     = cfg["engine"]
    log_cfg = cfg.get("logging", {}) or {}

    period = str(eng.get("period", "60d"))
    limit  = eng.get("limit", None)

    strat = make_strategy(strategy_name, cfg, tf)

    feed = CandleFeed(exchange=eng.get("exchange", "yahoo"), symbol=sym, timeframe=tf)
    df   = feed.fetch(period=period, limit=limit).sort_index()

    out_dir     = log_cfg.get("out_dir", "backtests")
    trades_path = os.path.join(out_dir, f"trades_{strategy_name}_{sym}_{tf}.csv")

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
            "strategy": strategy_name, "symbol": sym, "tf": tf,
            "trades_csv": trades_path, "final_balance": broker.balance,
            "closed": 0, "skipped": f"not enough candles (have={len(df)}, need>{warmup})",
        }

    for i in range(warmup, len(df) + 1):
        candle = df.iloc[i - 1]
        now    = candle.name

        broker.update_with_candle(candle, now, sym, tf)

        # 300 bars: covers atr_period*3 + swing_window*4 + 100 with room to spare.
        # Increase if swing_window > 20.
        context = df.iloc[max(0, i - 300):i]
        order   = strat.on_candles(context, sym)

        if order is None or order.side not in ("buy", "sell"):
            continue

        order.meta       = order.meta or {}
        order.meta["tf"] = tf

        broker.open_from_order(
            order=order,
            strategy=f"{strategy_name}_{tf}",
            now=now,
            market_close_price=to_float(candle["close"]),
        )

    last = df.iloc[-1]
    broker.force_close_all(
        now=last.name,
        market_close_price=to_float(last["close"]),
        symbol=sym,
        tf=tf,
        status="data_end",
    )

    if hasattr(strat, "print_diagnostics"):
        strat.print_diagnostics()

    return {
        "strategy": strategy_name, "symbol": sym, "tf": tf,
        "trades_csv": trades_path, "final_balance": broker.balance,
        "closed": len(broker.closed_positions), "skipped": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_trades_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=TRADES_HEADER)
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ("close_time", "open_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("pnl", "R"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.dropna(subset=["close_time"])


def _stats(df: pd.DataFrame, capital: float) -> dict:
    """Single source of truth for trade stats."""
    if df.empty:
        return {}

    total    = len(df)
    wins     = int((df["R"] > 0).sum())
    losses   = int((df["R"] <= 0).sum())
    wr       = wins / total
    avg_r    = float(df["R"].mean())
    avg_w    = float(df.loc[df["R"] > 0, "R"].mean()) if wins   else 0.0
    avg_l    = float(df.loc[df["R"] <= 0, "R"].mean()) if losses else 0.0
    expect   = wr * avg_w + (1 - wr) * avg_l
    total_pnl = float(df["pnl"].sum())
    ret_pct  = total_pnl / capital * 100

    max_dd = 0.0
    if "balance" in df.columns:
        bal    = df["balance"].astype(float)
        peak   = bal.cummax()
        max_dd = float(((bal - peak) / peak * 100).min())

    exits = df["status"].value_counts().to_dict()

    # Per-symbol breakdown
    sym_breakdown = {}
    if "symbol" in df.columns:
        for sym, sdf in df.groupby("symbol"):
            sw = int((sdf["R"] > 0).sum())
            sl = int((sdf["R"] <= 0).sum())
            sym_breakdown[sym] = {
                "trades": len(sdf),
                "wins": sw, "losses": sl,
                "wr": sw / len(sdf) * 100 if len(sdf) else 0,
                "avg_r": float(sdf["R"].mean()),
                "pnl": float(sdf["pnl"].sum()),
            }

    return {
        "total": total, "wins": wins, "losses": losses,
        "wr": wr * 100, "avg_r": avg_r, "avg_w": avg_w, "avg_l": avg_l,
        "expect": expect, "pnl": total_pnl, "ret_pct": ret_pct,
        "max_dd": max_dd, "exits": exits, "sym": sym_breakdown,
    }


def _sep(char="─", width=70):
    print(char * width)


def _print_strat_block(name: str, s: dict) -> None:
    exits_str = "  ".join(f"{k}:{v}" for k, v in sorted(s["exits"].items()))
    print(f"\n  Strategy : {name}")
    print(f"  Trades   : {s['total']}  (W:{s['wins']}  L:{s['losses']}  WR:{s['wr']:.0f}%)")
    print(f"  Avg R    : {s['avg_r']:+.2f}  |  Win:{s['avg_w']:+.2f}  Loss:{s['avg_l']:+.2f}")
    print(f"  Expect.  : {s['expect']:+.3f} R/trade")
    print(f"  PnL      : ${s['pnl']:+.2f}  ({s['ret_pct']:+.3f}% of allocated capital)")
    print(f"  Max DD   : {s['max_dd']:.2f}%")
    print(f"  Exits    : {exits_str}")
    if s["sym"]:
        print(f"  Per-symbol:")
        for sym, d in sorted(s["sym"].items()):
            print(f"    {sym:<6} trades={d['trades']:>2}  W:{d['wins']} L:{d['losses']}  "
                  f"WR:{d['wr']:.0f}%  avg_R:{d['avg_r']:+.2f}  pnl=${d['pnl']:+.2f}")


def compute_returns(results: list, starting_balance: float, strategies: list, symbols: list) -> None:
    n_syms   = max(1, len(symbols))
    n_strats = max(1, len(strategies))

    rows = []
    for r in results:
        if r.get("skipped"):
            continue
        tdf = _read_trades_csv(r["trades_csv"])
        if tdf.empty:
            continue
        tdf = tdf.copy()
        tdf["strategy_key"] = r["strategy"]
        rows.append(tdf)

    if not rows:
        print("\n=== No closed trades found ===")
        return

    trades = pd.concat(rows, ignore_index=True)
    trades["week"] = trades["close_time"].dt.to_period("W").astype(str)

    # ── Per-strategy deep stats ───────────────────────────────────────────
    _sep("═")
    print("  STRATEGY PERFORMANCE BREAKDOWN")
    _sep("═")
    strat_stats = {}
    for s in strategies:
        sub = trades[trades["strategy_key"] == s]
        capital = starting_balance * n_syms
        st = _stats(sub, capital)
        strat_stats[s] = st
        if st:
            _print_strat_block(s, st)

    # ── Side-by-side comparison table ────────────────────────────────────
    print()
    _sep("═")
    print(f"  {'STRATEGY':<14} {'N':>4} {'WR%':>5} {'AVG_R':>6} "
          f"{'EXPECT':>7} {'PNL$':>9} {'RET%':>7} {'MAXDD%':>7}")
    _sep()
    for s in strategies:
        st = strat_stats.get(s, {})
        if not st:
            print(f"  {s:<14}  — no trades —")
            continue
        print(f"  {s:<14} {st['total']:>4} {st['wr']:>5.0f} {st['avg_r']:>+6.2f} "
              f"{st['expect']:>+7.3f} {st['pnl']:>+9.2f} {st['ret_pct']:>+7.3f} {st['max_dd']:>+7.2f}")

    comb_capital = starting_balance * n_syms * n_strats
    comb = _stats(trades, comb_capital)
    if comb:
        _sep()
        print(f"  {'ALL COMBINED':<14} {comb['total']:>4} {comb['wr']:>5.0f} {comb['avg_r']:>+6.2f} "
              f"{comb['expect']:>+7.3f} {comb['pnl']:>+9.2f} {comb['ret_pct']:>+7.3f} {comb['max_dd']:>+7.2f}")
    _sep("═")

    # ── Weekly returns ────────────────────────────────────────────────────
    denom_strat = starting_balance * n_syms
    denom_all   = starting_balance * n_syms * n_strats
    weeks       = sorted(trades["week"].unique())
    pw          = trades.groupby(["strategy_key", "week"])["pnl"].sum().reset_index()

    print("\n=== Weekly Return % (per strategy) ===")
    for s in strategies:
        print(f"\n  -- {s} --")
        sub = pw[pw["strategy_key"] == s].set_index("week")["pnl"]
        for w in weeks:
            pnl = float(sub.get(w, 0.0))
            print(f"    {w} | pnl={pnl:+9.2f} | {pnl / denom_strat * 100:+7.3f}%")

    print("\n=== Weekly Return % (ALL combined) ===")
    pwa = trades.groupby("week")["pnl"].sum()
    for w in weeks:
        pnl = float(pwa.get(w, 0.0))
        print(f"  {w} | pnl={pnl:+9.2f} | {pnl / denom_all * 100:+7.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eng               = cfg["engine"]
    symbols           = eng.get("symbols", ["SPY", "QQQ", "GLD"])
    timeframes        = eng.get("timeframes", ["5m"])
    strategies_to_run = eng.get("strategies_to_run", ["swing_bos"])

    results = []
    print("\n=== Phase-1 batch backtests ===")
    for strat_name in strategies_to_run:
        print(f"\n--- Strategy: {strat_name} ---")
        for tf in timeframes:
            for sym in symbols:
                r = backtest_one(strat_name, sym, tf, cfg)
                results.append(r)
                if r["skipped"]:
                    print(f"  [SKIP] {sym} {tf} -> {r['skipped']}")
                else:
                    print(f"  [DONE] {sym} {tf} | closed={r['closed']} | final={r['final_balance']:.2f}")

    print("\n=== Run Summary ===")
    for r in results:
        tag = "SKIP" if r["skipped"] else "OK"
        print(f"  {tag:4} | {r['strategy']:14} | {r['symbol']:5} {r['tf']:3} | "
              f"closed={r['closed']:3} | final={r['final_balance']:.2f}")

    starting_balance = float(eng.get("paper_balance", 10000.0))
    compute_returns(results, starting_balance, strategies_to_run, symbols)
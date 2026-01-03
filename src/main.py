import yaml

from src.core.data import CandleFeed
from src.core.execution import PaperBroker
from src.strategies.liquidity_sweep import LiquiditySweep
from src.strategies.opening_range_breakout import OpeningRangeBreakout
from src.strategies.ema_trend_pullback import EMATrendPullback


def build_strategies(tf: str, symbol: str):
    """
    Build the list of strategies we want to run for a given symbol+timeframe.
    All three strategies are active so we can collect data for Phase 1.
    """
    strategies = []

    # ---------- Liquidity Sweep ----------
    strategies.append(
        LiquiditySweep(
            name=f"liq_sweep_{tf}",
            lookback=10,
            risk_reward=1.5,
        )
    )

    # ---------- Opening Range Breakout ----------
    # Map timeframe -> number of bars that form the opening range
    if tf == "5m":
        or_bars = 6   # first 30 minutes
    elif tf == "15m":
        or_bars = 4   # first hour
    elif tf == "30m":
        or_bars = 4   # first 2 hours
    else:
        or_bars = 4

    strategies.append(
        OpeningRangeBreakout(
            name=f"orb_{tf}",
            or_bars=or_bars,        # IMPORTANT: param name matches strategy code
            risk_reward=2.0,
            once_per_day=True,
        )
    )

    # ---------- EMA Trend Pullback ----------
    strategies.append(
        EMATrendPullback(
            name=f"ema_trend_{tf}",
            fast_ema=9,
            slow_ema=21,
            swing_lookback=5,
            max_distance_pct=0.0025,
            risk_reward=2.0,
        )
    )

    return strategies


def run_backtest_for(sym: str, tf: str, cfg: dict, broker: PaperBroker) -> None:
    """
    Run all strategies for a single symbol+timeframe on ONE shared broker.
    """
    eng = cfg["engine"]
    ex = eng["exchange"]

    # ---- 1) fetch data for this symbol+TF ----
    feed = CandleFeed(exchange=ex, symbol=sym, timeframe=tf)
    df = feed.fetch_latest(limit=500)
    print(f"\nFetched {len(df)} candles for {sym} at {tf}")

    if df.empty:
        print("No data, skipping.")
        return

    strategies = build_strategies(tf, sym)

    # Generic warmup so indicators / OR have enough history
    warmup = 50
    if len(df) <= warmup:
        print("Not enough candles after warmup, skipping.")
        return

    for i in range(warmup, len(df)):
        candle = df.iloc[i - 1]
        now = candle.name

        # 1) Update all open positions on this candle
        broker.update_with_candle(candle, now, sym, tf)

        # 2) Context = all candles up to this one (NO future data)
        context = df.iloc[:i]

        # 3) Let every strategy look at the same context and maybe fire
        for strat in strategies:
            signal = strat.on_candles(context, sym)
            if signal is not None and signal.side in ("buy", "sell"):
                # tag the order with timeframe so broker can filter positions correctly
                signal.meta = signal.meta or {}
                signal.meta["tf"] = tf

                broker.open_from_order(
                    order=signal,
                    strategy=strat.name,
                    now=now,
                    market_close_price=float(candle["close"]),
                )

    # Final update on the last candle for this symbol+TF
    last_candle = df.iloc[-1]
    broker.update_with_candle(last_candle, last_candle.name, sym, tf)

    # IMPORTANT for sequential backtests:
    # close remaining positions for this dataset at the last close
    broker.force_close_all(
        now=last_candle.name,
        market_close_price=float(last_candle["close"]),
        symbol=sym,
        tf=tf,
        status="data_end",
    )

    print(
        f"Finished {sym} {tf}. "
        f"Global balance: {broker.balance:.2f}, "
        f"Open positions: {len(broker.open_positions)}"
    )


if __name__ == "__main__":
    # ---- Load config ----
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eng = cfg["engine"]
    log_cfg = cfg.get("logging", {})

    symbols = eng["symbols"]
    timeframes = eng["timeframes"]

    # ---- Create ONE shared broker for all strategies/symbols/TFs ----
    broker = PaperBroker(
        starting_balance=float(eng.get("paper_balance", 10000.0)),
        risk_per_trade=float(eng.get("risk_per_trade", 0.005)),
        slippage_bps=float(eng.get("slippage_bps", 0.0)),
        trades_csv=log_cfg.get("trades_csv", "trades.csv"),
        max_loss_pct=float(eng.get("max_loss_pct", 0.03)),
    )

    # ---- Run backtests for each symbol+TF on the same broker ----
    for tf in timeframes:
        for sym in symbols:
            run_backtest_for(sym, tf, cfg, broker)

    print("\n=== Global account summary ===")
    print(f"Final balance: {broker.balance:.2f}")
    print(f"Total closed positions: {len(broker.closed_positions)}")
    print(f"Open positions remaining: {len(broker.open_positions)}")
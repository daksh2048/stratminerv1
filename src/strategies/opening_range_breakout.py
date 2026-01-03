import pandas as pd 

from src.core.types import Order
from .base import Strategy

class OpeningRangeBreakout(Strategy):
    """
    ORB- Opening Range Breakout is s a stretgy that identifies breakouts from the opening price range of a trading session. If the price breaks above the opening range, it generates a buy signal; if it breaks below, it generates a sell signal. The win rate of the strategy once there is a breakout is approximately 60-70%.
    """

    def __init__(self, name:str, **params) -> None:
        super().__init__(name, **params)
        self.last_trade_day = None

    def on_candles(self, df: pd.DataFrame, symbol: str) -> Order:
        # --- read params with defaults ---
        or_bars = int(self.params.get("or_bars", 6))          # how many candles define OR
        rr = float(self.params.get("risk_reward", 2.0))       # risk-reward multiple
        once_per_day = bool(self.params.get("once_per_day", True))

        # Need enough candles overall to even define an opening range + some trading candles
        if len(df) < or_bars + 2:
            return Order(symbol, None, None, None, None, "not enough data", {})

        # Last candle is the current decision candle
        last = df.iloc[-1]
        ts = last.name  # index is a Timestamp
        current_day = ts.normalize()  # midnight of that day (date without time)

        # --- restrict to today's candles only ---
        # df.index is a DatetimeIndex; normalize() gives dates for each row
        day_mask = df.index.normalize() == current_day
        day_df = df[day_mask]

        # If today's data doesn't have OR complete + at least one trading candle, skip
        if len(day_df) < or_bars + 1:
            return Order(symbol, None, None, None, None, "opening range not complete", {})

        # Position of the last candle within today's subset
        # day_df.index.get_loc(ts) returns the integer position for that timestamp
        pos_in_day = day_df.index.get_loc(ts)
        # If we're still inside the opening range candles, do not trade yet
        if pos_in_day < or_bars:
            return Order(symbol, None, None, None, None, "inside opening range", {})

        # If we only allow one trade per day and we've already traded today, skip
        if once_per_day and self.last_trade_day is not None and self.last_trade_day == current_day:
            return Order(symbol, None, None, None, None, "already traded this day", {})

        # --- compute opening range high/low ---
        opening_segment = day_df.iloc[:or_bars]
        or_high = float(opening_segment["high"].max())
        or_low = float(opening_segment["low"].min())

        last_open = float(last["open"])
        last_close = float(last["close"])

        # --- Long ORB: breakout above OR high ---
        if last_close > or_high and last_open <= or_high:
            entry = last_close
            stop = or_low  # opposite side of range
            if stop >= entry:
                return Order(symbol, None, None, None, None, "invalid long stop (OR too tight)", {})
            take = entry + rr * (entry - stop)

            self.last_trade_day = current_day
            return Order(
                symbol=symbol,
                side="buy",
                entry=entry,
                stop=stop,
                take=take,
                reason="ORB breakout long",
                meta={"or_high": or_high, "or_low": or_low},
            )

        # --- Short ORB: breakout below OR low ---
        if last_close < or_low and last_open >= or_low:
            entry = last_close
            stop = or_high
            if stop <= entry:
                return Order(symbol, None, None, None, None, "invalid short stop (OR too tight)", {})
            take = entry - rr * (stop - entry)

            self.last_trade_day = current_day
            return Order(
                symbol=symbol,
                side="sell",
                entry=entry,
                stop=stop,
                take=take,
                reason="ORB breakout short",
                meta={"or_high": or_high, "or_low": or_low},
            )

        # --- No setup this candle ---
        return Order(symbol, None, None, None, None, "no ORB setup", {})

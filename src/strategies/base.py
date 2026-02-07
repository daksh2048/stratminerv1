from abc import ABC, abstractmethod
import pandas as pd
from src.core.types import Order


class Strategy(ABC):
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def on_candles(self, df: pd.DataFrame, symbol:str) -> Order:
        ...

BaseStrategy = Strategy
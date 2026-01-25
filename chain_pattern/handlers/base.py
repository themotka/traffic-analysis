from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        self._next = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = self.process(df)
        if self._next:
            return self._next.handle(processed)
        return processed

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

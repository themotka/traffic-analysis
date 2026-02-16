"""
Базовый класс цепочки обработчиков данных (Chain of Responsibility).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Handler(ABC):
    """
    Абстрактный обработчик в цепочке.

    Каждый обработчик выполняет свою логику в process и передаёт данные следующему.
    """

    def __init__(self) -> None:
        self._next: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        """
        Устанавливает следующий обработчик в цепочке.

        handler — следующий обработчик.
        Возвращает переданный handler для цепочки вызовов.
        """
        self._next = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает данные и передаёт следующему обработчику.

        df — входной датафрейм.
        Возвращает обработанный датафрейм.
        """
        processed = self.process(df)
        if self._next:
            return self._next.handle(processed)
        return processed

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет обработку данных. Переопределяется в наследниках.

        df — входной датафрейм.
        Возвращает обработанный датафрейм.
        """
        ...

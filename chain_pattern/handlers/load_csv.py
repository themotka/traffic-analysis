"""
Обработчик загрузки данных из CSV-файла.
"""
import pandas as pd
from .base import Handler


class LoadCSVHandler(Handler):
    """
    Загружает данные из CSV-файла по указанному пути.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Читает CSV-файл и возвращает датафрейм.

        df — игнорируется (входной датафрейм пустой при первом вызове).
        Возвращает загруженный датафрейм.
        """
        return pd.read_csv(self._path)

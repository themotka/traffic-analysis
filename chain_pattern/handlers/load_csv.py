import pandas as pd
from .base import Handler


class LoadCSVHandler(Handler):
    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.read_csv(self._path)

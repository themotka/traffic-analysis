"""
Обработчик нормализации названий столбцов.
"""
from .base import Handler


class NormalizeColumnsHandler(Handler):
    """
    Нормализует названия столбцов: приведение к нижнему регистру, замена пробелов и запятых.
    """

    def process(self, df):
        """
        Приводит названия столбцов к единому формату.

        df — входной датафрейм.
        Возвращает датафрейм с обновлёнными названиями столбцов.
        """
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(",", "")
        )
        return df

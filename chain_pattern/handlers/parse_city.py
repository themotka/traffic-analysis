"""
Обработчик парсинга города из столбца «город».
"""
from .base import Handler


class ParseCityHandler(Handler):
    """
    Извлекает основной город из столбца «город» (первое значение до запятой).
    """

    def process(self, df):
        """
        Берёт первый город из списка и сохраняет в столбец city.

        df — входной датафрейм с столбцом «город».
        Возвращает датафрейм с новым столбцом city, удаляет столбец «город».
        """
        df["city"] = df["город"].apply(lambda x: x.split(",")[0].strip())
        df.drop(columns=["город"], inplace=True)
        return df

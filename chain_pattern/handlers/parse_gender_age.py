"""
Обработчик парсинга пола и возраста из столбца «пол_возраст».
"""
import re
from .base import Handler

COLUMN_GENDER_AGE = "пол_возраст"


class ParseGenderAgeHandler(Handler):
    """
    Извлекает пол (gender) и возраст (age) из столбца «пол_возраст», удаляет исходный столбец.
    """

    def process(self, df):
        """
        Парсит пол (1 — мужской, 0 — женский) и возраст из текста.

        df — входной датафрейм с столбцом «пол_возраст».
        Возвращает датафрейм с новыми столбцами gender и age.
        """
        def parse_gender(value: str) -> int:
            return 1 if "муж" in value.lower() else 0

        def parse_age(value: str) -> int:
            match = re.search(r"(\d+)\s*года", value)
            return int(match.group(1)) if match else -1

        df["gender"] = df[COLUMN_GENDER_AGE].apply(parse_gender)
        df["age"] = df[COLUMN_GENDER_AGE].apply(parse_age)

        df.drop(columns=[COLUMN_GENDER_AGE], inplace=True)
        return df

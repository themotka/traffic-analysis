"""
Обработчик парсинга зарплаты из столбца «зп».
"""
import re
import numpy as np
from .base import Handler


class ParseSalaryHandler(Handler):
    """
    Извлекает числовое значение зарплаты из текстового столбца «зп».
    """

    def process(self, df):
        """
        Извлекает цифры из строки зарплаты и сохраняет в столбец salary.

        df — входной датафрейм с столбцом «зп».
        Возвращает датафрейм с новым столбцом salary.
        """
        def parse_salary(value: str) -> int:
            digits = re.sub(r"[^\d]", "", value)
            return int(digits) if digits else 0

        df["salary"] = df["зп"].apply(parse_salary)
        return df
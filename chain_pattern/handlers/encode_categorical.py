"""
Обработчик кодирования категориальных признаков.
"""
from sklearn.preprocessing import LabelEncoder
from .base import Handler


class EncodeCategoricalHandler(Handler):
    """
    Кодирует категориальные столбцы числами с помощью LabelEncoder.
    """

    CATEGORICAL_COLUMNS = [
        "ищет_работу_на_должность:",
        "занятость",
        "график",
        "city",
        "авто",
    ]

    def process(self, df):
        """
        Заменяет категориальные значения на числовые коды.

        df — входной датафрейм.
        Возвращает датафрейм с закодированными категориальными столбцами.
        """
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
        return df

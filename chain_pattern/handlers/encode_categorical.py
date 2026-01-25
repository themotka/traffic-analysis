from sklearn.preprocessing import LabelEncoder
from .base import Handler


class EncodeCategoricalHandler(Handler):
    CATEGORICAL_COLUMNS = [
        "ищет_работу_на_должность:",
        "занятость",
        "график",
        "city",
        "авто",
    ]

    def process(self, df):
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
        return df

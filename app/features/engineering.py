"""
Feature engineering для классификатора.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..data.parsers import find_column


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает признаки для классификатора.

    Добавляет salary, age, experience_years, city_encoded, experience_text.
    """
    df = df.copy()
    df["salary"] = df["_salary"]
    df["age"] = df["_age"].replace(-1, df["_age"].median())
    df["experience_years"] = df["_exp_years"].replace(-1.0, df["_exp_years"].median())

    city_col = find_column(df, "Город", "город")
    df["city"] = (
        df[city_col]
        .fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .replace("", "не указан")
    )
    df["city_encoded"] = LabelEncoder().fit_transform(df["city"].astype(str))

    exp_col = find_column(df, "Опыт", "опыт")
    job_col = find_column(df, "должность", "Ищет работу на должность")
    df["experience_text"] = (
        df[exp_col].fillna("").astype(str).str[:1500]
        + " "
        + df[job_col].fillna("").astype(str)
    ).str.strip()

    return df

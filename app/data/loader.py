"""
Загрузка резюме из CSV-файлов hh.ru.
"""
from pathlib import Path

import pandas as pd


def load_resumes(csv_path: Path) -> pd.DataFrame:
    """
    Загружает резюме из CSV.

    csv_path — путь к файлу.
    Использует engine='python' и on_bad_lines='skip' для устойчивости к битым строкам.
    """
    return pd.read_csv(
        csv_path,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
    )

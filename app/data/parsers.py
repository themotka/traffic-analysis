"""
Парсинг полей резюме (зарплата, возраст, опыт).
"""
import re

import pandas as pd


def find_column(df: pd.DataFrame, *candidates: str) -> str:
    """Находит колонку по подстроке в названии."""
    for col in df.columns:
        col_lower = str(col).lower()
        for c in candidates:
            if c.lower() in col_lower:
                return col
    return df.columns[3]


def parse_salary(value) -> int:
    """Извлекает числовое значение зарплаты."""
    if pd.isna(value):
        return 0
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else 0


def parse_age(value) -> int:
    """Извлекает возраст из строки «Пол, возраст»."""
    if pd.isna(value):
        return -1
    match = re.search(r"(\d+)\s*(?:лет|год[ау]?)", str(value).lower())
    return int(match.group(1)) if match else -1


def parse_experience_years(value) -> float:
    """Извлекает опыт работы в годах из столбца «Опыт»."""
    if pd.isna(value):
        return -1.0
    text = str(value)[:200]
    patterns = [
        (r"Опыт работы\s+(\d+)\s+лет\s+(\d+)\s+месяц", lambda m: int(m.group(1)) + int(m.group(2)) / 12),
        (r"Опыт работы\s+(\d+)\s+лет", lambda m: float(m.group(1))),
        (r"Опыт работы\s+(\d+)\s+месяц", lambda m: int(m.group(1)) / 12),
    ]
    for pattern, handler in patterns:
        match = re.search(pattern, text)
        if match:
            return handler(match)
    return -1.0

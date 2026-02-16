"""
Создание целевой переменной (junior/middle/senior).
"""
import pandas as pd

from ..data.parsers import find_column, parse_age, parse_experience_years, parse_salary

JUNIOR_MAX_YEARS = 2
SENIOR_MIN_YEARS = 6


def assign_level_from_title(job_title) -> str | None:
    """Определяет уровень по должности. None если не удалось."""
    if pd.isna(job_title):
        return None
    title_lower = str(job_title).lower()
    if any(x in title_lower for x in ["junior", "джуниор", "стажёр", "стажер", "intern"]):
        return "junior"
    if any(x in title_lower for x in ["senior", "старший", "lead", "руководитель", "начальник"]):
        return "senior"
    return None


def assign_level_from_experience(years: float) -> str:
    """Определяет уровень по опыту работы."""
    if years < 0:
        return "middle"
    if years <= JUNIOR_MAX_YEARS:
        return "junior"
    if years >= SENIOR_MIN_YEARS:
        return "senior"
    return "middle"


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт целевую переменную level (junior/middle/senior).

    Приоритет: явное указание в должности > опыт.
    """
    job_col = find_column(df, "должность", "Ищет работу на должность")
    exp_col = find_column(df, "Опыт", "опыт")
    salary_col = find_column(df, "ЗП", "зп")
    age_col = find_column(df, "Пол", "возраст")

    df = df.copy()
    df["_salary"] = df[salary_col].apply(parse_salary)
    df["_age"] = df[age_col].apply(parse_age)
    df["_exp_years"] = df[exp_col].apply(parse_experience_years)

    levels = []
    for _, row in df.iterrows():
        from_title = assign_level_from_title(row[job_col])
        from_exp = assign_level_from_experience(row["_exp_years"])
        levels.append(from_title if from_title is not None else from_exp)
    df["level"] = levels
    return df

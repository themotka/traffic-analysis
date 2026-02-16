"""
Фильтрация IT-разработчиков из общего пула резюме.
"""
import pandas as pd

from .parsers import find_column

IT_KEYWORDS = [
    "программист", "разработчик", "developer", "devops",
    "backend", "frontend", "fullstack", "full-stack",
    "системный администратор", "sysadmin", "it инженер", "it-инженер",
    "web-программист", "web-разработчик", "data scientist",
    "python", "java", "javascript", "c++", "c#", "php",
    "руководитель разработки", "team lead", "tech lead",
    "архитектор", "web-администратор", "web-мастер",
    "инженер-программист", "техник-программист",
]


def is_it_developer(job_title) -> bool:
    """Проверяет, относится ли должность к IT-разработке."""
    if pd.isna(job_title):
        return False
    title_lower = str(job_title).lower()
    return any(kw.lower() in title_lower for kw in IT_KEYWORDS)


def filter_it_developers(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только резюме IT-разработчиков."""
    job_col = find_column(df, "должность", "Ищет работу на должность")
    return df[df[job_col].apply(is_it_developer)].copy()

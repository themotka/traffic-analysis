"""
Сборка пайплайна обработки данных из chain_pattern.
"""
from pathlib import Path
from handlers.load_csv import LoadCSVHandler
from handlers.normalize_columns import NormalizeColumnsHandler
from handlers.parse_gender_age import ParseGenderAgeHandler
from handlers.parse_salary import ParseSalaryHandler
from handlers.parse_city import ParseCityHandler
from handlers.encode_categorical import EncodeCategoricalHandler
from handlers.build_matrices import BuildMatricesHandler


def build_pipeline(csv_path: str):
    """
    Собирает цепочку обработчиков для подготовки данных из CSV.

    csv_path — путь к входному CSV-файлу.
    Возвращает первый обработчик цепочки (LoadCSVHandler).
    Матрицы сохраняются в папку с исходным файлом.
    """
    output_dir = Path(csv_path).parent

    loader = LoadCSVHandler(csv_path)
    loader \
        .set_next(NormalizeColumnsHandler()) \
        .set_next(ParseGenderAgeHandler()) \
        .set_next(ParseSalaryHandler()) \
        .set_next(ParseCityHandler()) \
        .set_next(EncodeCategoricalHandler()) \
        .set_next(BuildMatricesHandler(output_dir))

    return loader

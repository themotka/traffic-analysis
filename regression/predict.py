"""Предсказание зарплат по матрице признаков с помощью обученной модели."""

import logging
from pathlib import Path

import numpy as np

from .model_io import load_model

logger = logging.getLogger(__name__)


def load_x_data(path: Path) -> np.ndarray:
    """
    Загружает матрицу признаков из файла .npy (выход пайплайна chain_pattern).

    path — путь к файлу x_data.npy.

    Возвращает матрицу признаков (объекты по строкам, признаки по столбцам).
    Генерирует исключение при отсутствии файла или некорректном формате.
    """
    logger.debug("Загрузка признаков из %s", path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    try:
        X = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Не удалось загрузить данные из {path}: файл повреждён или не в формате .npy."
        ) from exc
    if X.ndim != 2:
        raise ValueError(
            f"Ожидается матрица признаков (2 измерения), получено {X.ndim} измерений."
        )
    logger.debug("Загружено объектов: %d, признаков: %d", X.shape[0], X.shape[1])
    return X


def predict_salaries(x_path: Path) -> list[float]:
    """
    Возвращает список предсказанных зарплат в рублях для объектов из x_data.npy.

    Загружает модель из regression/resources и матрицу признаков из указанного файла.
    Предсказания соответствуют порядку строк в x_data.npy.

    x_path — путь к файлу x_data.npy (выход пайплайна chain_pattern).

    Возвращает предсказанные зарплаты в рублях для каждого объекта.
    Генерирует исключение при отсутствии файлов или несовместимости признаков.
    """
    X = load_x_data(x_path)
    model = load_model()
    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Число признаков в данных ({X.shape[1]}) не совпадает с ожидаемым моделью ({model.n_features_in_}). "
            "Используйте данные, полученные тем же пайплайном chain_pattern."
        )
    pred = model.predict(X)
    result = [float(x) for x in pred]
    logger.debug("Получено предсказаний: %d", len(result))
    return result

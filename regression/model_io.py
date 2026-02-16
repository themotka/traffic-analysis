"""Загрузка и сохранение весов регрессионной модели в папке resources пакета regression."""

import logging
from pathlib import Path
from typing import Union

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODEL_FILENAME = "salary_model.joblib"


def _resources_dir() -> Path:
    """
    Возвращает абсолютный путь к папке resources внутри пакета regression.
    """
    return Path(__file__).resolve().parent / "resources"


def get_model_path() -> Path:
    """
    Возвращает путь к файлу весов модели в папке resources.

    Результат указывает на salary_model.joblib в regression/resources.
    """
    return _resources_dir() / MODEL_FILENAME


def load_model() -> Pipeline:
    """
    Загружает обученный пайплайн (преобразования + модель) из папки regression/resources.

    Возвращает обученный пайплайн с методом predict.
    Генерирует исключение при отсутствии файла, повреждении или несовместимости объекта.
    """
    path = get_model_path()
    logger.debug("Загрузка модели из %s", path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл весов модели не найден: {path}. "
            "Сначала выполните обучение: python -m regression.train <путь_к_папке_с_x_data_и_y_data>"
        )
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"Не удалось загрузить модель из {path}: файл повреждён или несовместим."
        ) from exc
    if not hasattr(pipeline, "predict") or not callable(getattr(pipeline, "predict")):
        raise ValueError(
            f"Загруженный объект не поддерживает предсказание: {type(pipeline).__name__}"
        )
    if not hasattr(pipeline, "n_features_in_"):
        raise ValueError(
            "Загруженный объект не имеет атрибута n_features_in_ (несовместимая версия модели)."
        )
    logger.debug("Модель успешно загружена")
    return pipeline


def save_model(pipeline: Union[Pipeline, object]) -> None:
    """
    Сохраняет обученный пайплайн в папку regression/resources.

    pipeline — обученный пайплайн (StandardScaler + регрессор) для сохранения.
    Генерирует исключение при сбое создания папки или записи файла.
    """
    resources = _resources_dir()
    path = resources / MODEL_FILENAME
    logger.debug("Сохранение модели в %s", path)
    try:
        resources.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path)
    except OSError as exc:
        raise OSError(
            f"Не удалось сохранить модель в {path}: {exc}"
        ) from exc
    logger.debug("Модель успешно сохранена")

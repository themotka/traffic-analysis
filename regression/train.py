"""Скрипт обучения регрессионной модели на выходе пайплайна chain_pattern."""

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import compute_metrics, log_metrics
from .model_io import save_model

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает матрицы признаков и целевых значений из папки с выходом пайплайна.

    Ожидаются файлы x_data.npy (признаки) и y_data.npy (зарплаты в рублях)
    в указанной папке.

    Args:
        data_dir: Путь к папке, содержащей x_data.npy и y_data.npy.

    Returns:
        Кортеж (X, y): матрица признаков и вектор целевых значений (зарплаты).

    Raises:
        FileNotFoundError: Отсутствует x_data.npy или y_data.npy в указанной папке.
        ValueError: Форма или тип данных в файлах не соответствуют ожидаемым.
    """
    x_path = data_dir / "x_data.npy"
    y_path = data_dir / "y_data.npy"
    logger.debug("Загрузка данных из %s", data_dir)
    if not x_path.is_file():
        raise FileNotFoundError(
            f"Файл признаков не найден: {x_path}. "
            "Укажите папку с выходом пайплайна chain_pattern (x_data.npy, y_data.npy)."
        )
    if not y_path.is_file():
        raise FileNotFoundError(
            f"Файл целевых значений не найден: {y_path}. "
            "Укажите папку с выходом пайплайна chain_pattern (x_data.npy, y_data.npy)."
        )
    try:
        X = np.load(x_path, allow_pickle=False)
        y = np.load(y_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Не удалось загрузить данные из {data_dir}: файлы повреждены или не в формате .npy."
        ) from exc
    if X.ndim != 2:
        raise ValueError(
            f"Ожидается матрица признаков (2 измерения), получено {X.ndim} измерений."
        )
    if y.ndim != 1:
        raise ValueError(
            f"Ожидается вектор целевых значений (1 измерение), получено {y.ndim} измерений."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Число строк в x_data.npy ({X.shape[0]}) не совпадает с длиной y_data.npy ({y.shape[0]})."
        )
    logger.debug("Загружено объектов: %d, признаков: %d", X.shape[0], X.shape[1])
    return X, y


def train(data_dir: Path) -> None:
    """
    Обучает регрессионный пайплайн и сохраняет его в папку regression/resources.

    Пайплайн: StandardScaler + GradientBoostingRegressor.
    Использует данные из указанной папки (x_data.npy, y_data.npy).
    Сохраняется в regression/resources/salary_model.joblib.

    Args:
        data_dir: Путь к папке с выходом пайплайна chain_pattern.

    Raises:
        FileNotFoundError: Отсутствует x_data.npy или y_data.npy.
        ValueError: Ошибка формата данных или обучение не удалось.
        OSError: Не удалось сохранить модель в regression/resources.
    """
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.debug(
        "Разбиение: обучающая выборка %d, тестовая %d",
        len(y_train),
        len(y_test),
    )
    logger.debug("Запуск обучения пайплайна: StandardScaler + GradientBoostingRegressor")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise ValueError(
            "Ошибка при обучении модели. Проверьте корректность данных (нет NaN/Inf, числовой тип)."
        ) from exc
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    metrics_train = compute_metrics(y_train, y_pred_train)
    metrics_test = compute_metrics(y_test, y_pred_test)
    log_metrics(metrics_train, prefix="Метрики на обучающей выборке: ")
    log_metrics(metrics_test, prefix="Метрики на тестовой выборке: ")
    pipeline.fit(X, y)
    save_model(pipeline)
    logger.debug("Обучение завершено, модель сохранена в regression/resources")


def main() -> None:
    """Точка входа CLI для обучения модели."""
    if len(sys.argv) != 2:
        print(
            "Использование: python -m regression.train <путь_к_папке_с_x_data_и_y_data>",
            file=sys.stderr,
        )
        sys.exit(1)
    path_arg = sys.argv[1]
    try:
        data_dir = Path(path_arg).resolve()
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Указанный путь не является папкой: {data_dir}")
        train(data_dir)
    except (FileNotFoundError, ValueError, OSError, NotADirectoryError) as exc:
        logger.debug("Ошибка при обучении: %s", exc)
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Вычисление метрик качества регрессии."""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Вычисляет метрики качества регрессии по фактическим и предсказанным значениям.

    Считаются MAE (средняя абсолютная ошибка, руб.), MSE (средняя квадратичная ошибка),
    RMSE (корень из MSE, руб.), R² (коэффициент детерминации).

    y_true — фактические значения целевой переменной (зарплаты в рублях).
    y_pred — предсказанные значения.

    Возвращает словарь с ключами "mae", "mse", "rmse", "r2". Значения в рублях, кроме R².
    Генерирует исключение при несовпадении длин или пустых массивах.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size != y_pred.size:
        raise ValueError(
            f"Длины массивов не совпадают: y_true {y_true.size}, y_pred {y_pred.size}."
        )
    if y_true.size == 0:
        raise ValueError("Массивы не должны быть пустыми.")
    logger.debug("Вычисление метрик по %d объектам", y_true.size)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """
    Логирует метрики на уровне DEBUG с опциональным префиксом.

    metrics — словарь метрик (результат compute_metrics).
    prefix — строка, добавляемая к сообщению (например, "Обучающая выборка: ").
    """
    for name, value in metrics.items():
        if name == "r2":
            logger.debug("%s%s = %.6f", prefix, name, value)
        else:
            logger.debug("%s%s = %.2f руб.", prefix, name, value)

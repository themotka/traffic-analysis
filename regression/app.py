"""
CLI-приложение: предсказание зарплат по файлу x_data.npy (выход пайплайна chain_pattern).

Интерфейс: python -m regression.app chain_pattern/x_data.npy из корня проекта
Вывод: список зарплат в рублях (по одному float на строку).
"""

import logging
import sys
from pathlib import Path

from .predict import predict_salaries

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Точка входа CLI.

    Читает путь к x_data.npy из аргумента командной строки, загружает модель
    из regression/resources, выводит предсказанные зарплаты (по одному значению на строку).
    Завершает работу с кодом 1 при неверных аргументах или ошибке предсказания.
    """
    if len(sys.argv) != 2:
        logger.debug("Неверное число аргументов: %d, ожидается 1", len(sys.argv) - 1)
        print(
            "Использование: python -m regression.app path/to/x_data.npy",
            file=sys.stderr,
        )
        sys.exit(1)
    x_path = Path(sys.argv[1]).resolve()
    try:
        salaries = predict_salaries(x_path)
    except (ValueError, OSError) as exc:
        logger.debug("Ошибка при предсказании: %s", exc)
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
    for s in salaries:
        print(s)


if __name__ == "__main__":
    main()

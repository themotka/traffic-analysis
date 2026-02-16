"""
Точка входа для пайплайна chain_pattern: подготовка данных из CSV.
"""
import sys
import pandas as pd
from pipeline import build_pipeline


def main():
    """
    Запускает пайплайн обработки CSV-файла.

    Ожидает один аргумент — путь к CSV-файлу.
    Генерирует исключение при неверном числе аргументов.
    """
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python main.py hh.csv")

    csv_path = sys.argv[1]

    pipeline = build_pipeline(csv_path)
    pipeline.handle(pd.DataFrame())


if __name__ == "__main__":
    main()

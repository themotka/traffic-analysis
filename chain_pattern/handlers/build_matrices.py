"""
Обработчик построения матриц признаков и целевых значений.
"""
import numpy as np
from pathlib import Path
from .base import Handler


class BuildMatricesHandler(Handler):
    """
    Формирует матрицы x_data.npy и y_data.npy и сохраняет в указанную папку.
    """

    def __init__(self, output_dir: Path):
        super().__init__()
        self._output_dir = output_dir

    def process(self, df):
        """
        Извлекает признаки и зарплаты, сохраняет в .npy-файлы.

        df — входной датафрейм с обработанными признаками и столбцом salary.
        Возвращает исходный датафрейм без изменений.
        """
        y = df["salary"].to_numpy(dtype=np.float32)
        x = df.drop(columns=["salary", "зп"]).select_dtypes(include=["number"]).to_numpy()

        np.save(self._output_dir / "x_data.npy", x)
        np.save(self._output_dir / "y_data.npy", y)

        return df

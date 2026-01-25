import numpy as np
from pathlib import Path
from .base import Handler


class BuildMatricesHandler(Handler):
    def __init__(self, output_dir: Path):
        super().__init__()
        self._output_dir = output_dir

    def process(self, df):
        y = df["salary"].to_numpy(dtype=np.float32)
        x = df.drop(columns=["salary", "зп"]).select_dtypes(include=["number"]).to_numpy()

        np.save(self._output_dir / "x_data.npy", x)
        np.save(self._output_dir / "y_data.npy", y)

        return df

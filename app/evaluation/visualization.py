"""
Визуализация результатов классификации.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import LABELS


def plot_class_balance(df: pd.DataFrame, output_path: Path) -> None:
    """Строит график баланса классов."""
    counts = df["level"].value_counts().reindex(LABELS).fillna(0)
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_xlabel("Уровень")
    axes[0].set_ylabel("Количество резюме")
    axes[0].set_title("Распределение по уровням")

    axes[1].pie(
        counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90
    )
    axes[1].set_title("Доля классов")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

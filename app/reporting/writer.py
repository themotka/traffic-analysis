"""
Формирование и запись отчётов о классификации.
"""
from pathlib import Path

import pandas as pd

from ..evaluation.metrics import LABELS


def format_conclusions(
    df: pd.DataFrame,
    report: dict,
    cv_report: dict,
    feature_importance: dict,
    use_smote: bool,
) -> str:
    """Формирует текстовые выводы о качестве модели."""
    lines = []
    counts = df["level"].value_counts()

    lines.append("1. Дисбаланс классов:")
    for lev in LABELS:
        n = counts.get(lev, 0)
        pct = 100 * n / len(df) if len(df) > 0 else 0
        lines.append(f"   - {lev}: {n} ({pct:.1f}%)")
    if counts.min() > 0 and counts.max() / counts.min() > 3:
        lines.append("   Применены class_weight='balanced' и SMOTE.")

    lines.append("\n2. Важность признаков (топ-10):")
    for feat, val in sorted(feature_importance.items(), key=lambda x: -x[1])[:10]:
        short = str(feat)[:50] + "..." if len(str(feat)) > 50 else feat
        lines.append(f"   - {short}: {val:.3f}")

    lines.append("\n3. Улучшения: class_weight, TF-IDF, 5-fold CV" + (", SMOTE" if use_smote else ""))

    return "\n".join(lines)


def write_report(
    output_path: Path, report_str: str, cv_report_str: str, conclusions: str
) -> None:
    """Записывает отчёт в файл."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report (test split)\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_str)
        f.write("\n\nCross-Validation (5-fold stratified)\n")
        f.write("=" * 50 + "\n\n")
        f.write(cv_report_str)
        f.write("\n\nВыводы:\n")
        f.write(conclusions)

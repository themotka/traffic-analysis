"""
CLI для запуска PoC классификации IT-разработчиков.

Запуск: python -m app.cli
"""
import sys
from pathlib import Path

from .data.filters import filter_it_developers
from .data.loader import load_resumes
from .evaluation.metrics import compute_report
from .evaluation.visualization import plot_class_balance
from .features.engineering import prepare_features
from .features.labeling import create_target_variable
from .model.pipeline import NUMERIC_COLUMNS, TEXT_COLUMN
from .model.training import train_and_evaluate
from .reporting.writer import format_conclusions, write_report


def run_poc(csv_path: Path, output_dir: Path) -> dict:
    """
    Выполняет полный цикл PoC: загрузка, обучение, отчёт.

    csv_path — путь к CSV.
    output_dir — папка для сохранения результатов.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_resumes(csv_path)
    df = filter_it_developers(df)
    df = create_target_variable(df)
    df = prepare_features(df)

    if len(df) < 5:
        raise ValueError(f"Недостаточно IT-резюме: {len(df)}. Нужно минимум 5.")

    plot_class_balance(df, output_dir / "class_balance.png")

    X = df[NUMERIC_COLUMNS + [TEXT_COLUMN]].copy()
    X[NUMERIC_COLUMNS] = X[NUMERIC_COLUMNS].fillna(0)
    y = df["level"]

    result = train_and_evaluate(X, y)
    conclusions = format_conclusions(
        df,
        result["report"],
        result["cv_report"],
        result["feature_importance"],
        result["use_smote"],
    )

    report_str = compute_report(result["y_test"], result["y_pred"])
    cv_report_str = compute_report(y, result["cv_pred"])

    report_path = output_dir / "classification_report.txt"
    write_report(report_path, report_str, cv_report_str, conclusions)

    print("Test split:\n", report_str)
    print("\nCross-Validation (5-fold):\n", cv_report_str)
    print("\nГрафик:", output_dir / "class_balance.png")
    print("Отчёт:", report_path)

    return {
        "n_samples": len(df),
        "report": result["report"],
        "cv_report": result["cv_report"],
        "feature_importance": result["feature_importance"],
    }


def main() -> None:
    """Точка входа CLI."""
    project_root = Path(__file__).resolve().parent.parent
    csv_path = None
    for name in ("hh.csv", "hh_medium.csv", "hh_small.csv"):
        p = project_root / name
        if p.is_file():
            csv_path = p
            break

    if csv_path is None:
        print("Ошибка: не найден hh.csv, hh_medium.csv или hh_small.csv", file=sys.stderr)
        sys.exit(1)

    output_dir = project_root / "output"
    print(f"Используется датасет: {csv_path.name}")
    run_poc(csv_path, output_dir)


if __name__ == "__main__":
    main()

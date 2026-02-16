"""
Обучение и оценка модели.
"""
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from ..evaluation.metrics import compute_report, get_feature_importance
from .pipeline import build_pipeline


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Обучает модель и вычисляет метрики.

    Возвращает словарь: pipeline, report, cv_report, feature_importance, use_smote,
    y_test, y_pred, cv_pred.
    """
    counts = y.value_counts()
    min_count = int(counts.min()) if len(counts) > 0 else 0
    use_smote = min_count >= 3
    stratify_arg = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_arg
    )

    pipeline = build_pipeline(use_smote)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    cv_pred = cross_val_predict(pipeline, X, y, cv=5, method="predict")

    report = compute_report(y_test, y_pred, as_dict=True)
    cv_report = compute_report(y, cv_pred, as_dict=True)
    feature_importance = get_feature_importance(pipeline)

    return {
        "pipeline": pipeline,
        "report": report,
        "cv_report": cv_report,
        "feature_importance": feature_importance,
        "use_smote": use_smote,
        "y_test": y_test,
        "y_pred": y_pred,
        "cv_pred": cv_pred,
    }

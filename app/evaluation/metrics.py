"""
Вычисление метрик классификации.
"""
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

LABELS = ["junior", "middle", "senior"]


def compute_report(y_true, y_pred, as_dict: bool = False):
    """Единая функция для формирования classification report."""
    return classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=as_dict,
        zero_division=0,
    )


def get_feature_importance(pipeline: Pipeline) -> dict:
    """Извлекает важность признаков из RandomForest."""
    clf = pipeline.named_steps["classifier"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return dict(zip(names, clf.feature_importances_))

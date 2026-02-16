"""
Построение ML-пайплайна для классификации.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = ["salary", "age", "experience_years", "city_encoded"]
TEXT_COLUMN = "experience_text"


def build_pipeline(use_smote: bool) -> Pipeline:
    """Собирает пайплайн: препроцессинг + опционально SMOTE + классификатор."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("text", TfidfVectorizer(max_features=50, min_df=1, ngram_range=(1, 2)), TEXT_COLUMN),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=42
    )

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            return ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(k_neighbors=2, random_state=42)),
                    ("classifier", clf),
                ],
                memory=None,
            )
        except ImportError:
            pass

    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", clf)],
        memory=None,
    )

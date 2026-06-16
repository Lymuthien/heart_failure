import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from heart_failure.features import ConjRuleFeature
from heart_failure.config.features import CONJUNCTIVE_RULES


class ConjRuleBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, conditions: dict, min_count: int = 50, min_target_rate: float = 0.9):
        self.transformer = ConjRuleFeature(conditions, min_count, min_target_rate)

    def fit(self, X: pd.DataFrame, y) -> "ConjRuleBaseline":
        self.transformer.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame)-> np.ndarray:
        X_transformed = self.transformer.transform(X)
        y = X_transformed[CONJUNCTIVE_RULES].to_numpy()

        return y


class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, remaining_columns: list[str] = None):
        self.remaining_columns = remaining_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.remaining_columns:
            X = X[self.remaining_columns]

        return X


def build_lr_pipeline(fe_pipeline: Pipeline, rem_columns: list[str] = None, random_state: int = 42) -> Pipeline:
    pipeline = Pipeline(
        [
            ("feature_engineering", fe_pipeline),
            ("column_selector", ColumnSelector(rem_columns)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=random_state))
        ]
    )

    return pipeline

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

from heart_failure.features import ConjRuleFeature
from heart_failure.config.features import CONJUNCTIVE_RULES
from heart_failure.config.modeling import CB_EVAL_METRIC


class ConjRuleBaseline(BaseEstimator, ClassifierMixin):
    def __init__(
        self, conditions: dict, min_count: int = 50, min_target_rate: float = 0.9
    ):
        self.transformer = ConjRuleFeature(conditions, min_count, min_target_rate)

    def fit(self, X: pd.DataFrame, y) -> "ConjRuleBaseline":
        self.transformer.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
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


class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cat_features=None, **params):
        self.cat_features = cat_features
        self._params = params.copy()

    def __getattr__(self, name):
        if "model_" in self.__dict__:
            return getattr(self.model_, name)
        raise AttributeError(name)

    def get_params(self, deep=True):
        params = {"cat_features": self.cat_features, **self._params}

        return params

    def set_params(self, **params):
        if "cat_features" in params:
            self.cat_features = params.pop("cat_features")

        if params:
            self._params.update(params)
            if hasattr(self, "model_"):
                self.model_.set_params(**params)

        return self

    def fit(self, X: pd.DataFrame, y):
        self.model_ = CatBoostClassifier(**self._params)
        self.model_.fit(X, y, cat_features=self.cat_features)

        self.classes_ = self.model_.classes_
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame):
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict_proba(X)


def build_lr_pipeline(
    fe_pipeline: Pipeline, rem_columns: list[str] = None, random_state: int = 42
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("feature_engineering", fe_pipeline),
            ("column_selector", ColumnSelector(rem_columns)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=random_state)),
        ]
    )

    return pipeline


def build_dt_pipeline(
    fe_pipeline: Pipeline, rem_columns: list[str] = None, random_state: int = 42
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("feature_engineering", fe_pipeline),
            ("column_selector", ColumnSelector(rem_columns)),
            ("model", DecisionTreeClassifier(random_state=random_state)),
        ]
    )

    return pipeline


def build_catboost_pipeline(random_state: int = 42, cat_features=None) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "model",
                CatBoostWrapper(
                    cat_features=cat_features,
                    random_seed=random_state,
                    eval_metric=CB_EVAL_METRIC,
                ),
            ),
        ]
    )

    return pipeline

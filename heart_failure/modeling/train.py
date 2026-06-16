import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from heart_failure.config.features import CONJUNCTIVE_RULES
from heart_failure.features import ConjRuleFeature


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
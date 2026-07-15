import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from heart_failure.config.config import AGE, MAX_HR
from heart_failure.config.features import (
    TE_FEATURES,
    BINARY_CAT_FEATURES,
    CONJ_RULES,
    CONJ_MIN_MASK_COUNT,
    CONJ_MIN_TARGET_RATE,
    CONJUNCTIVE_RULES,
    F_BINS,
    TE_FEATURE_CONFIG,
)
from heart_failure.config.train import TE_CV, RANDOM_STATE


class CrossTargetEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        features: list[str],
        bins: dict[str, int] = None,
        merge_rules: dict[str, str] = None,
        cv=None,
    ):
        self.features = features
        self.bins = bins
        self.merge_rules = merge_rules
        self.cv = cv

    def _encode(self, X: pd.DataFrame, y=None, fit=False):
        X = X.copy()

        if fit and self.bins:
            for col, n_bins in self.bins.items():
                binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
                binner.fit(X[[col]])
                self.binners_[col] = binner

        cross = self._make_cross(X)

        if fit:
            self.encoder_ = TargetEncoder(cv=self.cv)
            encoded = self.encoder_.fit_transform(cross.to_frame("cross"), y)
        else:
            encoded = self.encoder_.transform(cross.to_frame("cross"))

        name = "_".join(self.features) + "_te"
        X[name] = encoded.ravel()
        return X

    def fit(self, X: pd.DataFrame, y):
        self.binners_ = {}
        self._encode(X, y=y, fit=True)
        return self

    def transform(self, X: pd.DataFrame):
        return self._encode(X, fit=False)

    def fit_transform(self, X: pd.DataFrame, y):
        self.binners_ = {}
        return self._encode(X, y, fit=True)

    def _make_cross(self, X: pd.DataFrame):
        parts = []

        for col in self.features:
            if col in self.binners_:
                values = (
                    self.binners_[col]
                    .transform(X[[col]])
                    .astype(int)
                    .ravel()
                    .astype(str)
                )
            else:
                values = X[col].astype(str).values
            parts.append(pd.Series(values, index=X.index))

        cross = parts[0]

        for part in parts[1:]:
            cross = cross + "__" + part

        if self.merge_rules:
            cross = cross.replace(self.merge_rules)

        return cross


class ConjRuleFeature(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        conditions: dict,
        min_count: int = 50,
        min_target_rate: float = 0.9,
        combine_rules: bool = True,
    ):
        self.conditions = conditions
        self.min_count = min_count
        self.min_target_rate = min_target_rate
        self.combine_rules = combine_rules

    def fit(self, X: pd.DataFrame, y):
        cond_values = {name: rule(X) for name, rule in self.conditions.items()}
        rules = []

        for left_name, right_name in combinations(self.conditions.keys(), 2):
            mask = cond_values[left_name] & cond_values[right_name]

            count = mask.sum()
            if count < self.min_count:
                continue

            target_rate = y.loc[mask].mean()
            if target_rate < self.min_target_rate:
                continue

            rules.append(((left_name, right_name), count, target_rate))

        self.selected_rules_ = [rule_names for rule_names, _, _ in rules]

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        cond_values = {name: rule(X) for name, rule in self.conditions.items()}
        feature = np.zeros(len(X), dtype=int)

        for left_name, right_name in self.selected_rules_:
            values = (cond_values[left_name] & cond_values[right_name]).astype(int)
            if self.combine_rules:
                feature |= values
            else:
                X[f"{left_name} & {right_name}"] = values

        if self.combine_rules:
            X[CONJUNCTIVE_RULES] = feature

        return X


class RatioFeature(BaseEstimator, TransformerMixin):
    def __init__(self, numerator: str, denominator: str):
        self.numerator = numerator
        self.denominator = denominator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[f"{self.numerator}_{self.denominator}_ratio"] = (
            X[self.numerator] / X[self.denominator]
        )
        return X


class FixedBinsDiscretizer(TransformerMixin, BaseEstimator):
    def __init__(self, bins: list[int]):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_enc = X.apply(lambda s: pd.cut(s, bins=self.bins, labels=False))
        return X_enc

    def set_output(self, *, transform=None):
        return self


class Preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, cv, n_bins=8, use_binner=True, rename_features=True):
        self.cv = cv
        self.n_bins = n_bins
        self.use_binner = use_binner
        self.rename_features = rename_features

    # noinspection PyTypeChecker
    def _build_transformer(self):
        transformers = [
            ("target_encoder", TargetEncoder(cv=self.cv), TE_FEATURES),
            ("binary_encoder", OrdinalEncoder(), BINARY_CAT_FEATURES),
        ]

        if self.use_binner:
            for feature, bins in F_BINS.items():
                binner = FixedBinsDiscretizer(bins)
                transformers.append((f"binner_{feature}", binner, [feature]))

        transformer = ColumnTransformer(
            transformers,
            remainder="passthrough",
            verbose_feature_names_out=self.rename_features,
        )
        transformer.set_output(transform="pandas")
        return transformer

    def fit(self, X, y=None):
        self.transformer_ = self._build_transformer()
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)


def _build_te_features(cv):
    steps = []
    for config in TE_FEATURE_CONFIG:
        steps.append(
            (
                config["name"],
                CrossTargetEncoder(
                    features=config["features"],
                    bins=config["bins"],
                    merge_rules=config["merge_rules"],
                    cv=cv,
                ),
            )
        )

    return steps


def get_fe_pipeline(
    cv=None,
    combine_rules=True,
    conj_feature=True,
    n_bins=8,
    use_binner=True,
    rename_features=False,
) -> Pipeline:
    if cv is None:
        cv = StratifiedKFold(n_splits=TE_CV, shuffle=True, random_state=RANDOM_STATE)

    steps = []
    if conj_feature:
        rule_aggregator = ConjRuleFeature(
            CONJ_RULES, CONJ_MIN_MASK_COUNT, CONJ_MIN_TARGET_RATE, combine_rules
        )
        steps.append(("rule_aggregator", rule_aggregator))

    steps += [
        ("age_maxhr_ratio", RatioFeature(AGE, MAX_HR)),
        *_build_te_features(cv),
        ("preprocessor", Preprocessor(cv, n_bins, use_binner, rename_features)),
    ]

    pipeline = Pipeline(steps)

    return pipeline

import numpy as np
import pandas as pd
from itertools import product, combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from heart_failure.config.config import (
    SEX,
    AGE,
    MAX_HR,
    OLDPEAK,
    ST_SLOPE,
    CHEST_PAIN_TYPE,
    RESTING_ECG,
)
from heart_failure.config.features import (
    TE_FEATURES,
    QBINNED_FEATURES,
    BINARY_CAT_FEATURES,
    MAX_HR_TE_FEATURES,
    CONJ_RULES,
    CONJ_MIN_MASK_COUNT,
    CONJ_MIN_TARGET_RATE,
    SEX_TE_FEATURES,
    CONJUNCTIVE_RULES,
    TE_CV,
    RANDOM_STATE,
    ST_TE_FEATURES,
    SEX_CAT_TE,
    F_BINS,
    RECG_CAT_TE,
)


class TEByBins(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        binning_cols: list[str],
        cat_cols: list[str],
        n_bins: list[int] | int = 5,
        cv=None,
    ):
        self.binning_cols = binning_cols
        self.cat_cols = cat_cols
        self.n_bins = n_bins
        self.cv = cv

    def _encode(self, X: pd.DataFrame, y=None, fit=False):
        X = X.copy()
        if isinstance(self.n_bins, int):
            self.n_bins = [self.n_bins] * len(self.binning_cols)

        bin_labels = []
        for bin_col, n_bins in zip(self.binning_cols, self.n_bins):
            if fit:
                binner = KBinsDiscretizer(n_bins, encode="ordinal")
                bins = binner.fit_transform(X[[bin_col]]).ravel().astype(int)
                self.binners_[bin_col] = binner
            else:
                binner = self.binners_[bin_col]
                bins = binner.transform(X[[bin_col]]).ravel().astype(int)

            bin_labels += [pd.Series(bins, index=X.index).astype(str)]

        for cat_col in self.cat_cols:
            df = pd.concat([X[cat_col].astype(str)] + bin_labels, axis=1)
            cross = df.agg("__".join, axis=1)

            encoder = self.encoders_[cat_col]
            if fit:
                encoded = encoder.fit_transform(cross.to_frame("cross"), y)
            else:
                encoded = encoder.transform(cross.to_frame("cross"))

            X[f"{"_".join(self.binning_cols)}_{cat_col}_te"] = encoded.ravel()

        return X

    def _init_encoders(self):
        return {cat_col: TargetEncoder(cv=self.cv) for cat_col in self.cat_cols}

    def fit(self, X: pd.DataFrame, y):
        self.binners_ = {}
        self.encoders_ = self._init_encoders()

        self._encode(X, y=y, fit=True)
        return self

    def transform(self, X: pd.DataFrame):
        return self._encode(X, fit=False)

    def fit_transform(self, X: pd.DataFrame, y):
        self.binners_ = {}
        self.encoders_ = self._init_encoders()

        return self._encode(X, y=y, fit=True)


class CrossTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols_groups: tuple[list[str], list[str]], cv=None):
        self.cat_cols_groups = cat_cols_groups
        self.cv = cv

    def _encode(self, X: pd.DataFrame, y=None, fit=False):
        X = X.copy()

        for left_col, right_col in product(*self.cat_cols_groups):
            cross = X[left_col].astype(str) + "__" + X[right_col].astype(str)

            encoder = self.encoders_[(left_col, right_col)]
            if fit:
                encoded = encoder.fit_transform(cross.to_frame("cross"), y)
            else:
                encoded = encoder.transform(cross.to_frame("cross"))

            X[f"{left_col}_{right_col}_te"] = encoded.ravel()

        return X

    def _init_encoders(self):
        return {
            (left_col, right_col): TargetEncoder(cv=self.cv)
            for left_col, right_col in product(*self.cat_cols_groups)
        }

    def fit(self, X: pd.DataFrame, y):
        self.encoders_ = self._init_encoders()
        self._encode(X, y=y, fit=True)
        return self

    def transform(self, X: pd.DataFrame):
        return self._encode(X, fit=False)

    def fit_transform(self, X: pd.DataFrame, y):
        self.encoders_ = self._init_encoders()
        return self._encode(X, y=y, fit=True)


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
            binner = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal")
            transformers.append(("binner_quantile", binner, QBINNED_FEATURES))

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
        ("oldpeak_cpt_te", TEByBins([OLDPEAK], [CHEST_PAIN_TYPE], n_bins=4, cv=cv)),
        ("st_te", TEByBins(ST_TE_FEATURES, [ST_SLOPE], n_bins=[4, 3], cv=cv)),
        ("te_by_max_hr", TEByBins([MAX_HR], MAX_HR_TE_FEATURES, n_bins=4, cv=cv)),
        ("sex_te", TEByBins(SEX_TE_FEATURES, [SEX], n_bins=[5, 3], cv=cv)),
        ("sex_cat_te", CrossTargetEncoder((SEX_CAT_TE, [SEX]), cv=cv)),
        ("recg_cat_te", CrossTargetEncoder((RECG_CAT_TE, [RESTING_ECG]), cv=cv)),
        ("oldpeak_recg_te", TEByBins([OLDPEAK], [RESTING_ECG], n_bins=4, cv=cv)),
        ("preprocessor", Preprocessor(cv, n_bins, use_binner, rename_features)),
    ]

    pipeline = Pipeline(steps)

    return pipeline


def get_prep_pipeline(
    cv=None, n_bins=8, use_binner=True, rename_features=False
) -> Pipeline:
    if cv is None:
        cv = StratifiedKFold(n_splits=TE_CV, shuffle=True, random_state=RANDOM_STATE)

    return Pipeline(
        [("preprocessor", Preprocessor(cv, n_bins, use_binner, rename_features))]
    )

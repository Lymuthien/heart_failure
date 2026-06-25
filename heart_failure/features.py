import numpy as np
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from itertools import product, combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from heart_failure.config.config import SEX, AGE, MAX_HR, OLDPEAK, FASTING_BS
from heart_failure.config.features import (
    TE_FEATURES,
    BINARIZED_FEATURES,
    BINARY_CAT_FEATURES,
    Z_SCORED_FEATURES,
    OLDPEAK_TE_FEATURES,
    MAX_HR_TE_FEATURES,
    FASTING_BS_TE_FEATURES,
    CONJ_RULES,
    CONJ_MIN_MASK_COUNT,
    CONJ_MIN_TARGET_RATE,
    SEX_TE_FEATURES,
    CONJUNCTIVE_RULES,
    TE_CV,
    RANDOM_STATE,
)


class GroupZScore(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols: list[str], value_cols: list[str]):
        self.group_cols = group_cols
        self.value_cols = value_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.stats_ = X.groupby(self.group_cols)[self.value_cols].agg(["mean", "std"])
        self.stats_.columns = [f"{col}_{stat}" for col, stat in self.stats_.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        tmp = X.merge(
            self.stats_, left_on=self.group_cols, right_index=True, how="left"
        )

        for col in self.value_cols:
            z = (tmp[col] - tmp[f"{col}_mean"]) / tmp[f"{col}_std"]
            X[f"{col}_z"] = z.replace([np.inf, -np.inf], 0).fillna(0)

        return X


class TargetEncoderByBins(BaseEstimator, TransformerMixin):
    def __init__(
        self, binning_cols: list[str], cat_cols: list[str], n_bins: list[int] | int = 5, cv=None
    ):
        self.binning_cols = binning_cols
        self.cat_cols = cat_cols
        self.n_bins = n_bins
        self.cv = cv

    def _encode(self, X: pd.DataFrame, y=None, fit=False):
        X = X.copy()
        if isinstance(self.n_bins, int):
            self.n_bins = [self.n_bins] * len(self.binning_cols)

        for bin_col, n_bins in zip(self.binning_cols, self.n_bins):
            if fit:
                binner = KBinsDiscretizer(n_bins, encode="ordinal")
                bins = binner.fit_transform(X[[bin_col]]).ravel().astype(int)
                self.binners_[bin_col] = binner
            else:
                binner = self.binners_[bin_col]
                bins = binner.transform(X[[bin_col]]).ravel().astype(int)

            bin_labels = pd.Series(bins, index=X.index)

            for cat_col in self.cat_cols:
                cross = X[cat_col].astype(str) + "__" + bin_labels.astype(str)

                encoder = self.encoders_[(bin_col, cat_col)]
                if fit:
                    encoded = encoder.fit_transform(cross.to_frame("cross"), y)
                else:
                    encoded = encoder.transform(cross.to_frame("cross"))

                X[f"{bin_col}_{cat_col}_te"] = encoded.ravel()

        return X

    def _init_encoders(self):
        return {
            (bin_col, cat_col): TargetEncoder(cv=self.cv)
            for bin_col, cat_col in product(self.binning_cols, self.cat_cols)
        }

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


def get_preprocessor(cv):
    binarizer = KBinsDiscretizer(n_bins=10, encode="ordinal")
    preprocessor = ColumnTransformer(
        [
            ("target_encoder", TargetEncoder(cv=cv), TE_FEATURES),
            ("binarizer", binarizer, BINARIZED_FEATURES),
            ("binary_encoder", OrdinalEncoder(), BINARY_CAT_FEATURES),
        ],
        remainder="passthrough",
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def get_fe_pipeline(cv=None, combine_rules=True) -> Pipeline:
    if cv is None:
        cv = StratifiedKFold(n_splits=TE_CV, shuffle=True, random_state=RANDOM_STATE)

    pipeline = Pipeline(
        [
            ("age_maxhr_ratio", RatioFeature(AGE, MAX_HR)),
            (
                "rule_aggregator",
                ConjRuleFeature(
                    CONJ_RULES, CONJ_MIN_MASK_COUNT, CONJ_MIN_TARGET_RATE, combine_rules
                ),
            ),
            ("group_zscore", GroupZScore([AGE, SEX], Z_SCORED_FEATURES)),
            (
                "oldpeak_te",
                TargetEncoderByBins([OLDPEAK], OLDPEAK_TE_FEATURES, n_bins=4, cv=cv),
            ),
            (
                "te_by_max_hr",
                TargetEncoderByBins([MAX_HR], MAX_HR_TE_FEATURES, n_bins=4, cv=cv),
            ),
            ("sex_te", TargetEncoderByBins(SEX_TE_FEATURES, [SEX], n_bins=5, cv=cv)),
            (
                "fasting_bs_te",
                CrossTargetEncoder(([FASTING_BS], FASTING_BS_TE_FEATURES), cv=cv),
            ),
            ("preprocessor", get_preprocessor(cv)),
        ]
    )

    return pipeline


def select_by_vif(
    X: pd.DataFrame,
    y: pd.Series,
    cols: list = None,
    threshold: int = 10,
    max_r_diff: float = 0.01,
) -> tuple[list, pd.Series]:
    remaining_cols = cols.copy() if cols else X.columns.to_list()
    drop_order = []
    last_dropped = None
    last_r = 0

    while True:
        values = X[remaining_cols].values
        res = sm.OLS(y, sm.add_constant(values)).fit()
        if last_r - res.rsquared_adj > max_r_diff:
            remaining_cols.append(last_dropped)
            break

        last_r = res.rsquared_adj
        vifs = [vif(values, i) for i in range(len(remaining_cols))]
        vifs_max = max(vifs)
        if vifs_max <= threshold:
            break

        worst_idx = np.argmax(vifs)
        last_dropped = remaining_cols.pop(worst_idx)
        drop_order.append((last_dropped, vifs_max))

    return remaining_cols, pd.Series(dict(drop_order))

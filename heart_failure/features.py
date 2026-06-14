import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer


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
            X[f"{col}_z"] = (tmp[col] - tmp[f"{col}_mean"]) / tmp[f"{col}_std"]

        return X


class TargetEncoderByBins(BaseEstimator, TransformerMixin):
    def __init__(
        self, binning_cols: list[str], cat_cols: list[str], n_bins: int = 5, cv=None
    ):
        self.binning_cols = binning_cols
        self.cat_cols = cat_cols
        self.n_bins = n_bins
        self.cv = cv

    def _encode(self, X, y=None, fit=False):
        X = X.copy()

        for bin_col in self.binning_cols:
            if fit:
                binner = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal")
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

    def fit(self, X, y):
        self.binners_ = {}
        self.encoders_ = self._init_encoders()

        self._encode(X, y=y, fit=True)
        return self

    def transform(self, X):
        return self._encode(X, fit=False)

    def fit_transform(self, X, y):
        self.binners_ = {}
        self.encoders_ = self._init_encoders()

        return self._encode(X, y=y, fit=True)


class CrossTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols_groups: tuple[list[str], list[str]], cv=None):
        self.cat_cols_groups = cat_cols_groups
        self.cv = cv

    def _encode(self, X, y=None, fit=False):
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

    def fit(self, X, y):
        self.encoders_ = self._init_encoders()
        self._encode(X, y=y, fit=True)
        return self

    def transform(self, X):
        return self._encode(X, fit=False)

    def fit_transform(self, X, y):
        self.encoders_ = self._init_encoders()
        return self._encode(X, y=y, fit=True)


def add_features_ratio(
    df: pd.DataFrame, numerator: str, denominator: str
) -> pd.DataFrame:
    df = df.copy()

    df[f"{numerator}_{denominator}_ratio"] = df[numerator] / df[denominator]
    return df

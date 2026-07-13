import pandas as pd
import numpy as np
from scipy.stats import f_oneway


def target_pct_table_by_cat(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    temp = df.melt(id_vars=target, value_vars=cat_cols)
    pct_table = pd.crosstab(
        temp[target], [temp["variable"], temp["value"]], normalize="columns"
    )

    return pct_table


def cat_pct_distr_by_group(df: pd.DataFrame, group) -> pd.DataFrame:
    tables = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        ct = pd.crosstab(group, df[col], normalize="index")
        ct.columns = pd.MultiIndex.from_product([[col], ct.columns])
        tables.append(ct)

    return pd.concat(tables, axis=1)


def describe_by_group(df: pd.DataFrame, group):
    numeric = df.select_dtypes(include=["number"]).columns
    grouped = df.groupby(group, observed=True)

    result = grouped[numeric].agg(["mean", "median"])
    count = grouped.size().rename(("count", ""))
    result = pd.concat([count, result], axis=1)

    p_values = {}

    for col in numeric:
        samples = [g[col].dropna() for _, g in grouped]
        if any(len(s) < 2 for s in samples):
            p = np.nan
        else:
            _, p = f_oneway(*samples)
        p_values[(col, "mean")] = p

    p_values = pd.Series(p_values)
    result.loc["p_value"] = p_values

    return result

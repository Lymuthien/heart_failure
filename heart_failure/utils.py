import pandas as pd


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

import pandas as pd


def target_pct_table_by_cat(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    temp = df.melt(id_vars=target, value_vars=cat_cols)
    pct_table = pd.crosstab(
        temp[target], [temp["variable"], temp["value"]], normalize="columns"
    )

    return pct_table

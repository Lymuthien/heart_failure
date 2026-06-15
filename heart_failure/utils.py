import mlflow
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
)


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


def get_metrics(y_true, y_pred, y_proba=None) -> pd.Series:
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f2_score": fbeta_score(y_true, y_pred, beta=2),
    }
    if y_proba:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    return pd.Series(metrics)


def log_metrics(metrics: pd.Series, prefix: str = "") -> None:
    for name, metric in metrics.items():
        mlflow.log_metric(f"{prefix}{name}", float(metric))

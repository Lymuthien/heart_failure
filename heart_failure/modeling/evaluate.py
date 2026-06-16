import mlflow
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    precision_recall_curve
)


def find_best_threshold(y_true, y_proba, beta=1) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    best_idx = f.argmax()

    return round(thresholds[best_idx], 3)


def get_metrics(y_true, y_pred, y_proba=None) -> pd.Series:
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f2_score": fbeta_score(y_true, y_pred, beta=2),
    }
    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    return pd.Series(metrics)


def log_metrics(metrics: pd.Series, prefix: str = "") -> None:
    for name, metric in metrics.items():
        mlflow.log_metric(f"{prefix}{name}", float(metric))\


def optuna_cv_results_to_df(study: optuna.Study) -> pd.DataFrame:
    records = []

    for t in study.trials:
        scores = t.user_attrs.get("cv_scores")
        if scores is None:
            continue

        rec = {
            "params": t.params,
            "std_test_score": float(np.std(scores)),
            "mean_test_score": t.value,
        }
        for i, s in enumerate(scores):
            rec[f"split_test_{i}"] = s
        records.append(rec)

    df = pd.DataFrame(records)
    ascending = study.direction == optuna.study.StudyDirection.MINIMIZE
    df["rank_test_score"] = (
        df["mean_test_score"].rank(method="min", ascending=ascending).astype(int)
    )

    return df
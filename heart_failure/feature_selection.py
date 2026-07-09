import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from heart_failure.modeling.train import prec_scorer


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


def find_best_corr_threshold(X: pd.DataFrame, y: pd.Series, thresholds: list):
    corr_df = X.merge(y, left_index=True, right_index=True).corr()
    target_name = y.name
    result = []

    for thr in thresholds:
        cols = corr_df[corr_df[target_name].abs() >= thr].index.tolist()
        cols.remove(target_name)

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        score = cross_val_score(model, X[cols], y, cv=5, scoring=prec_scorer)

        result.append({
            "threshold": thr, "features": len(cols), "score": score.mean(), "std": score.std(),
        })
    result = pd.DataFrame(result).sort_values("score", ascending=False)
    return result.iloc[0]["threshold"], result
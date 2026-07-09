import pandas as pd


def predict_by_threshold(model, X: pd.DataFrame, threshold: float = 0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def _fit_score(estimator, X, y, sample_weight, train, test, scorer):
    if isinstance(estimator, Pipeline):
        param = {"model__sample_weight": sample_weight.iloc[train]}
    else:
        param = {"sample_weight": sample_weight.iloc[train]}
    estimator.fit(X.iloc[train], y.iloc[train], **param)

    result = scorer(
        estimator, X.iloc[test], y.iloc[test], sample_weight=sample_weight.iloc[test]
    )

    return result


def cross_val_score_sw(estimator, X, y, sample_weight, scoring, cv, n_jobs=1):
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(
            clone(estimator),
            X,
            y,
            sample_weight,
            train=train,
            test=test,
            scorer=scoring,
        )
        for train, test in cv.split(X, y)
    )

    return np.asarray(scores)

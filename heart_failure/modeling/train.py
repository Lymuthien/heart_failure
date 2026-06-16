import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from typing import Optional, Union

from heart_failure.features import get_fe_pipeline
from heart_failure.modeling.models import build_lr_pipeline


def optuna_search_lr(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    scoring: Union[str, callable] = "accuracy",
    n_jobs: int = -1,
    timeout: Optional[int] = None,
    rem_columns: list[str] = None,
    random_state: int = 42,
    fe_cv=None
) -> tuple[optuna.Study, Pipeline]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fe_pipeline = get_fe_pipeline(fe_cv)

    def objective(trial: optuna.trial.Trial) -> float:
        solver = trial.suggest_categorical("model__solver", ["lbfgs", "liblinear"])

        params = {
            "C": trial.suggest_float("model__C", 1e-4, 10, log=True),
            "solver": solver,
            "max_iter": trial.suggest_int("model__max_iter", 100, 500),
            "tol": trial.suggest_float("model__tol", 1e-5, 1e-3, log=True),
            "dual": False,
            "class_weight": "balanced",
        }

        if solver == "lbfgs":
            params["l1_ratio"] = 0
        else:
            params["l1_ratio"] = trial.suggest_int("model__l1_ratio", 0, 1)

        pipeline = build_lr_pipeline(fe_pipeline, rem_columns, random_state)
        pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        trial.set_user_attr("cv_scores", cv_scores.tolist())

        return cv_scores.mean()

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    best_pipeline = build_lr_pipeline(fe_pipeline, rem_columns, random_state)
    best_pipeline.set_params(**study.best_params)
    best_pipeline.fit(X, y)

    return study, best_pipeline

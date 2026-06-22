import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import average_precision_score


def ap_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)

    if proba.ndim == 2:
        proba = proba[:, 1]

    return average_precision_score(y, proba)


def optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_builder: callable,
    param_space: callable,
    scoring=ap_scorer,
    n_trials: int = 50,
    n_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    timeout=None,
    optimize_direction="maximize",
    **kwargs,
):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        model_params = param_space(trial)

        pipeline = model_builder(random_state=random_state, **kwargs)
        pipeline.set_params(**{f"model__{k}": v for k, v in model_params.items()})

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        trial.set_user_attr("cv_scores", scores.tolist())

        return scores.mean()

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction=optimize_direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    best_pipeline = model_builder(random_state=random_state, **kwargs)
    best_pipeline.set_params(**study.best_params)
    best_pipeline.fit(X, y)

    return study, best_pipeline


def lr_space(trial):
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

    return params


def dt_space(trial):
    params = {
        "max_depth": trial.suggest_int("model__max_depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 5, 30),
        "max_features": trial.suggest_categorical(
            "model__max_features", ["sqrt", "log2", None]
        ),
        "splitter": "best",
        "criterion": "gini",
    }

    return params


def catboost_space(trial):
    params = {
        "iterations": trial.suggest_int("model__iterations", 200, 800),
        "learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 0.2, log=True),
        "depth": trial.suggest_int("model__depth", 2, 5),
        "l2_leaf_reg": trial.suggest_float("model__l2_leaf_reg", 1e-4, 10, log=True),
        "min_data_in_leaf": trial.suggest_int("model__min_data_in_leaf", 5, 50),
        "subsample": trial.suggest_float("model__subsample", 0.7, 1),
        "rsm": trial.suggest_float("model__rsm", 0.5, 1),
        "verbose": 0,
    }

    return params


def xgboost_space(trial):
    params = {
        "n_estimators": trial.suggest_int("model__n_estimators", 200, 800),
        "learning_rate": trial.suggest_float(
            "model__learning_rate", 1e-3, 0.1, log=True
        ),
        "gamma": trial.suggest_float("model__gamma", 0, 2),
        "max_depth": trial.suggest_int("model__max_depth", 2, 5),
        "reg_lambda": trial.suggest_float("model__reg_lambda", 1e-4, 10, log=True),
        "reg_alpha": trial.suggest_float("model__reg_alpha", 1e-4, 1, log=True),
        "subsample": trial.suggest_float("model__subsample", 0.7, 1),
        "verbose": 0,
    }

    return params

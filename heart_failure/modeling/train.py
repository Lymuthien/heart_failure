import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold


def optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_builder: callable,
    param_space: callable,
    scoring="accuracy",
    n_trials: int = 50,
    n_splits: int = 5,
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
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

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

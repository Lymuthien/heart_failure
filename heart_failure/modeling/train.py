import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import average_precision_score, precision_score

from heart_failure.config.modeling import MIN_RECALL, FE_STEP_NAME, COL_SEL_STEP_NAME
from heart_failure.modeling.evaluate import find_best_threshold


def ap_scorer(estimator, X, y, sample_weight=None):
    proba = estimator.predict_proba(X)

    if proba.ndim == 2:
        proba = proba[:, 1]

    return average_precision_score(y, proba, sample_weight=sample_weight)


def prec_scorer(estimator, X, y, sample_weight=None):
    proba = estimator.predict_proba(X)
    if proba.ndim == 2:
        proba = proba[:, 1]

    threshold = find_best_threshold(y, proba, MIN_RECALL)
    pred = (proba >= threshold).astype(int)

    return precision_score(y, pred, sample_weight=sample_weight)


def optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_builder: callable,
    param_space: callable,
    scoring=prec_scorer,
    n_trials: int = 50,
    n_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    timeout=None,
    optimize_direction="maximize",
    cv_scorer=cross_val_score,
    cv_params=None,
    **kwargs,
):
    cv_params = cv_params or {}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        model_params = param_space(trial)

        pipeline = model_builder(random_state=random_state, **kwargs)
        pipeline.set_params(**model_params)

        scores = cv_scorer(pipeline, X, y, cv=cv, scoring=scoring, **cv_params)
        trial.set_user_attr("cv_scores", scores.tolist())

        return scores.mean()

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction=optimize_direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    best_pipeline = model_builder(random_state=random_state, **kwargs)
    params = {k: v for k, v in study.best_params.items() if "__" in k}
    best_pipeline.set_params(**params)
    best_pipeline.fit(X, y)

    return study, best_pipeline


def lr_space(trial):
    solver = trial.suggest_categorical("model__solver", ["lbfgs", "liblinear"])

    params = {
        "model__C": trial.suggest_float("model__C", 1e-4, 10, log=True),
        "model__solver": solver,
        "model__max_iter": trial.suggest_int("model__max_iter", 100, 500),
        "model__tol": trial.suggest_float("model__tol", 1e-5, 1e-3, log=True),
        "model__dual": False,
        "model__class_weight": "balanced",
    }

    if solver == "lbfgs":
        params["model__l1_ratio"] = 0
    else:
        params["model__l1_ratio"] = trial.suggest_int("model__l1_ratio", 0, 1)

    return params


def dt_space(trial):
    params = {
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 8),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 5, 30),
        "model__max_features": trial.suggest_categorical(
            "model__max_features", ["sqrt", "log2", None]
        ),
        "model__splitter": "best",
        "model__criterion": "gini",
    }

    return params


def catboost_space(trial):
    params = {
        "model__iterations": trial.suggest_int("model__iterations", 200, 800),
        "model__learning_rate": trial.suggest_float(
            "model__learning_rate", 1e-3, 0.2, log=True
        ),
        "model__depth": trial.suggest_int("model__depth", 2, 5),
        "model__l2_leaf_reg": trial.suggest_float(
            "model__l2_leaf_reg", 1e-4, 10, log=True
        ),
        "model__min_data_in_leaf": trial.suggest_int("model__min_data_in_leaf", 5, 50),
        "model__subsample": trial.suggest_float("model__subsample", 0.7, 1),
        "model__rsm": trial.suggest_float("model__rsm", 0.5, 1),
        "model__verbose": 0,
    }

    return params


def xgboost_space(trial):
    use_binner_key = f"{FE_STEP_NAME}__preprocessor__use_binner"
    top_k_key = f"{COL_SEL_STEP_NAME}__top_k"

    params = {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 800),
        "model__learning_rate": trial.suggest_float(
            "model__learning_rate", 1e-3, 0.1, log=True
        ),
        "model__gamma": trial.suggest_float("model__gamma", 0, 2),
        "model__max_depth": trial.suggest_int("model__max_depth", 2, 5),
        "model__reg_lambda": trial.suggest_float(
            "model__reg_lambda", 1e-4, 10, log=True
        ),
        "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-4, 1, log=True),
        "model__subsample": trial.suggest_float("model__subsample", 0.7, 1),
        "model__verbose": 0,
        use_binner_key: trial.suggest_categorical(use_binner_key, [True, False]),
        top_k_key: trial.suggest_int(top_k_key, 3, 20),
    }

    return params

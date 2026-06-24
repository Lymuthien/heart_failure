import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from typing import Callable


def _plot_cat_grid(
    df: pd.DataFrame,
    plot_func: Callable[[pd.DataFrame, str, plt.Axes], None],
    n_cols: int = 3,
    figsize_factor: int = 4,
):
    cat_cols = df.select_dtypes(include=["category", "object"]).columns

    n_rows = int(np.ceil(len(cat_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_factor * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols):
        plot_func(df, col, ax)

    for ax in axes[len(cat_cols) :]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def plot_cat_value_counts(df: pd.DataFrame, n_cols: int = 3, figsize_factor: int = 4):
    def plot_func(df, col, ax):
        df[col].value_counts(dropna=False).plot.bar(ax=ax)
        ax.set_ylabel("Count")

    _plot_cat_grid(df, plot_func, n_cols, figsize_factor)


def plot_cat_target_distribution(
    df: pd.DataFrame, target: str, n_cols: int = 3, figsize_factor: int = 4
):
    def plot_func(df, col, ax):
        pd.crosstab(df[col], df[target], dropna=False).plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Count")

    _plot_cat_grid(df, plot_func, n_cols, figsize_factor)


def plot_corr_matrix(
    corr_df: pd.DataFrame, max_abs: float = 1, width=500, height=500, fontsize=12
):
    fig = px.imshow(
        corr_df,
        labels=dict(x="feature", y="feature", color="corr"),
        x=corr_df.columns,
        y=corr_df.columns,
        color_continuous_scale="RdBu",
        zmin=-max_abs,
        zmax=max_abs,
    )
    fig.update_layout(width=width, height=height, font=dict(size=fontsize))
    fig.show()


def plot_pr_curve(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.show()


def kdeplot_features_by_target(df: pd.DataFrame, cols: list[str], target: str):
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))

    for ax, f in zip(axes, cols):
        sns.kdeplot(
            df, x=f, hue=target, fill=True, common_norm=False, alpha=0.3, ax=ax
        )

    plt.tight_layout()
    plt.show()
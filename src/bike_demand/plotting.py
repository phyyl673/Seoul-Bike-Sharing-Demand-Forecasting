from __future__ import annotations

from pathlib import Path

import dalex as dx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Global default style
sns.set_theme(style="whitegrid", context="talk")


def savefig(save_path: str | Path | None, *, fig: plt.Figure | None = None) -> None:
    """
    Save a matplotlib figure to disk if a path is provided.

    Parameters
    ----------
    save_path : str or Path or None
        Output path. If None, nothing is saved.
    fig : matplotlib.figure.Figure or None, optional
        Figure to save. If None, uses the current figure (`plt.gcf()`).

    Notes
    -----
    Parent directories are created if needed. The figure is closed after
    saving to avoid memory leaks when generating many plots.
    """
    if save_path is None:
        return

    fig = plt.gcf() if fig is None else fig
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Plots for EDA
# ---------------------------------------------------------------------
def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "rented_bike_count",
    *,
    bins: int = 50,
    color: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the distribution of the target variable (histogram + boxplot).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, optional
        Name of the target column. Defaults to "rented_bike_count".
    bins : int, optional
        Number of histogram bins. Defaults to 50.
    color : str or None, optional
        Color for histogram and boxplot. If None, seaborn default is used.
    save_path : str or Path or None, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Raises
    ------
    KeyError
        If `target_col` is not in the dataframe.
    """
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in dataframe.")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    sns.histplot(df[target_col], bins=bins, kde=True, ax=ax1, color=color)
    ax1.set_title(f"Distribution of {target_col}", fontweight="bold")
    ax1.set_ylabel("Frequency")

    sns.boxplot(x=df[target_col], ax=ax2, color=color)
    ax2.set_xlabel(target_col)

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_combined_explanatory_distributions(
    df: pd.DataFrame,
    columns: str | list[str],
    *,
    bins: int = 30,
    kde: bool = True,
    titles: list[str] | None = None,
    color: str | None = None,
    save_path: str | Path | None = None,
    ncols: int = 2,
) -> plt.Figure:
    """
    Plot distributions (histogram + boxplot) for one or more numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : str or list[str]
        Column(s) to plot.
    bins : int, optional
        Histogram bins. Defaults to 30.
    kde : bool, optional
        Whether to overlay KDE on histograms. Defaults to True.
    titles : list[str] or None, optional
        Optional titles per variable (must align with `columns` length).
    color : str or None, optional
        Color used for histograms/boxplots. If None, seaborn default is used.
    save_path : str or Path or None, optional
        If provided, saves the figure.
    ncols : int, optional
        Number of columns in the grid. Defaults to 2.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Raises
    ------
    KeyError
        If any requested column is not present in the dataframe.
    ValueError
        If `titles` length does not match number of columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    if titles is not None and len(titles) != len(columns):
        raise ValueError("If provided, `titles` must match the length of `columns`.")

    n_vars = len(columns)
    nrows = (n_vars + ncols - 1) // ncols

    fig = plt.figure(figsize=(10 * ncols, 6 * nrows))

    outer_gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        hspace=0.4,
        wspace=0.25,
        top=0.96,
        bottom=0.04,
        left=0.10,
        right=0.90,
    )

    for idx, column in enumerate(columns):
        row = idx // ncols
        col = idx % ncols

        inner_gs = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[row, col],
            height_ratios=[3, 1],
            hspace=0.05,
        )

        ax1 = fig.add_subplot(inner_gs[0])
        sns.histplot(data=df, x=column, bins=bins, kde=kde, ax=ax1, color=color)
        ax1.set_title(titles[idx] if titles else f"Distribution of {column}", pad=8)
        ax1.set_ylabel("Frequency")
        ax1.set_xlabel("")
        ax1.tick_params(labelbottom=False, bottom=False)

        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
        sns.boxplot(data=df, x=column, ax=ax2, color=color)
        ax2.set_xlabel(column)

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_categorical_frequency(
    df: pd.DataFrame,
    col: str | list[str] | list[dict],
    *,
    order: list | None = None,
    palette: str | list | None = None,
    title: str | list[str] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (6, 4),
):
    """
    Plot frequency distributions for one or more categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str, list[str], or list[dict]
        - str: single column name
        - list[str]: multiple column names
        - list[dict]: list of dicts with keys: 'col', and optionally
          'order', 'title', 'palette'
    order : list, optional
        Category order (single column only).
    palette : str, list, or None, optional
        Palette for plots. If None, seaborn default is used.
        For multiple plots, you may also specify per-column palette via dict.
    title : str or list[str], optional
        Title(s) for plot(s).
    ax : matplotlib.axes.Axes, optional
        Axis to draw onto (single column only).
    save_path : str or Path or None, optional
        If provided, saves the figure.
    ncols : int, optional
        Number of columns for grid when plotting multiple variables. Defaults to 3.
    figsize_per_plot : tuple[float, float], optional
        Size per subplot when plotting multiple variables. Defaults to (6, 4).

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.figure.Figure
        Axis for single plot, or Figure for multiple plots.
    """
    # Single column
    if isinstance(col, str):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.figure

        sns.countplot(
            data=df,
            x=col,
            order=order,
            palette=palette,
            ax=ax,
        )

        ax.set_ylabel("Count")
        ax.set_xlabel(col)

        if title:
            ax.set_title(title if isinstance(title, str) else str(title), fontweight="bold")

        fig.tight_layout()
        savefig(save_path, fig=fig)
        return ax

    # Multiple columns
    if len(col) == 0:
        raise ValueError("`col` must not be empty.")

    if isinstance(col[0], str):
        columns = [{"col": c} for c in col]  # type: ignore[index]
    else:
        columns = col  # type: ignore[assignment]

    if isinstance(title, str):
        titles = [title]
    elif isinstance(title, list):
        titles = title
    else:
        titles = None

    n_plots = len(columns)
    nrows = (n_plots + ncols - 1) // ncols

    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    for idx, config in enumerate(columns):
        ax_current = axes_flat[idx]

        col_name = config["col"]
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found in dataframe.")

        col_order = config.get("order", None)
        col_title = config.get("title", None)
        col_palette = config.get("palette", palette)

        if col_title is None and titles is not None and idx < len(titles):
            col_title = titles[idx]

        sns.countplot(
            data=df,
            x=col_name,
            order=col_order,
            palette=col_palette,
            ax=ax_current,
        )

        ax_current.set_ylabel("Count")
        ax_current.set_xlabel(col_name)

        if col_title:
            ax_current.set_title(col_title, fontweight="bold")

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_target_vs_continuous(
    df: pd.DataFrame,
    features: list[str],
    *,
    target: str = "rented_bike_count",
    color: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Scatter plots of target vs continuous features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        Continuous feature names to plot.
    target : str, optional
        Target column name. Defaults to "rented_bike_count".
    color : str or None, optional
        Point color. If None, seaborn default is used.
    save_path : str or Path or None, optional
        If provided, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    n_plots = len(features)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=True)
    axes = np.array(axes).ravel()

    for ax, feat in zip(axes, features):
        sns.scatterplot(
            data=df,
            x=feat,
            y=target,
            ax=ax,
            s=40,
            alpha=0.25,
            linewidth=0,
            color=color,
        )
        ax.set_title(f"{target} vs {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel(target)

    for ax in axes[len(features) :]:
        ax.set_visible(False)

    fig.suptitle("Rented Bike Count vs Continuous Features", fontsize=16)
    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_target_vs_categorical_mean(
    df: pd.DataFrame,
    features: list[str],
    *,
    target: str = "rented_bike_count",
    orders: dict[str, list] | None = None,
    ncols: int = 2,
    palette: str | list | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Bar plots of mean target value by categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        Categorical feature names to plot.
    target : str, optional
        Target column name. Defaults to "rented_bike_count".
    orders : dict[str, list] or None, optional
        Optional ordering for categories per feature name.
    ncols : int, optional
        Number of columns in grid. Defaults to 2.
    palette : str, list, or None, optional
        Palette passed to seaborn.barplot. If None, seaborn default.
    save_path : str or Path or None, optional
        If provided, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    nrows = (len(features) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharey=True,
    )
    axes = np.array(axes).ravel()

    for i, feat in enumerate(features):
        sns.barplot(
            data=df,
            x=feat,
            y=target,
            estimator="mean",
            order=orders.get(feat) if orders else None,
            palette=palette,
            ax=axes[i],
        )
        axes[i].set_title(f"Average {target} by {feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(target)

    for ax in axes[len(features) :]:
        ax.set_visible(False)

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_hourly_trend_by_season(
    df: pd.DataFrame,
    *,
    palette: str | list | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot average rentals by hour, with separate lines for each season.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must contain 'hour', 'seasons', 'rented_bike_count').
    palette : str, list, or None, optional
        Palette passed to seaborn.lineplot. If None, seaborn default.
    save_path : str or Path or None, optional
        If provided, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    hourly_avg = (
        df.groupby(["hour", "seasons"], observed=True)["rented_bike_count"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.lineplot(
        data=hourly_avg,
        x="hour",
        y="rented_bike_count",
        hue="seasons",
        style="seasons",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette=palette,
        ax=ax,
    )

    ax.set_title("Average Bike Rentals by Hour and Season", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_xticks(range(0, 24))
    ax.legend(title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    exclude_cols: list[str] | None = None,
    cmap: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 10),
    annot: bool = True,
) -> pd.DataFrame:
    """
    Plot a correlation heatmap for numeric features and return the matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str or None, optional
        If provided and present, show this column first in the matrix.
    exclude_cols : list[str] or None, optional
        Columns to exclude before computing correlations.
    cmap : str or None, optional
        Colormap for the heatmap. If None, seaborn default.
    save_path : str or Path or None, optional
        If provided, saves the figure.
    figsize : tuple[int, int], optional
        Figure size. Defaults to (14, 10).
    annot : bool, optional
        Whether to annotate correlation values. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Correlation matrix used for plotting.
    """
    data = df.copy()
    if exclude_cols:
        data = data.drop(columns=exclude_cols, errors="ignore")

    numeric_df = data.select_dtypes(include=["number"]).copy()

    if target_col is not None and target_col in numeric_df.columns:
        cols = [target_col] + [c for c in numeric_df.columns if c != target_col]
        numeric_df = numeric_df[cols]

    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    sns.heatmap(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        annot=annot,
        fmt=".2f",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return corr


# ---------------------------------------------------------------------
# Plots for Evaluation
# ---------------------------------------------------------------------
def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    color: str | None = None,
    line_color: str | None = "black",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot predicted values against actual values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    title : str
        Plot title (e.g. "LGBM: Predicted vs Actual (Validation)").
    color : str or None, optional
        Scatter color. If None, seaborn default is used.
    line_color : str or None, optional
        Color for the 45-degree reference line. Defaults to "black".
    save_path : str or Path or None, optional
        If provided, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.25, edgecolor=None, ax=ax, color=color)

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=line_color, linewidth=2)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 10,
    title: str,
    color: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot permutation feature importance as a horizontal bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Dataframe with at least ['feature', 'importance'] columns.
    top_n : int, optional
        Number of top features to display. Defaults to 10.
    title : str
        Plot title.
    color : str or None, optional
        Bar color. If None, matplotlib default is used.
    save_path : str or Path or None, optional
        If provided, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["feature"], df["importance"], color=color)
    ax.set_xlabel("Increase in loss after permutation")
    ax.set_title(title)

    fig.tight_layout()
    savefig(save_path, fig=fig)
    return fig


def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    *,
    model_name: str = "model",
    ax: plt.Axes | None = None,
    color: str | None = None,
    palette: str | list | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure | None:
    """
    Plot a partial dependence profile for a single feature.

    Notes
    -----
    - If `ax` is provided, the plot is drawn on that axis and the function
      returns None (and does not save).
    - If `ax` is None, the function creates a new figure and returns it;
      if `save_path` is provided, it also saves the figure.

    Parameters
    ----------
    model
        Fitted model exposing a `.predict()` method.
    X : pd.DataFrame
        Feature matrix used for profiling.
    y : pd.Series
        Target values (needed by dalex Explainer).
    feature : str
        Feature name to plot.
    model_name : str, optional
        Label used in dalex outputs. Defaults to "model".
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw onto. If None, a new figure is created.
    color : str or None, optional
        Line color for numeric PDP, or bar color for categorical PDP.
        If None, defaults are used.
    palette : str, list, or None, optional
        Palette for categorical PDP (levels). If None, defaults are used.
        If `color` is provided, it takes precedence for bars.
    save_path : str or Path or None, optional
        If provided and `ax` is None, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure if created internally, otherwise None.

    Raises
    ------
    ValueError
        If `feature` is not in `X`.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    # Numeric feature: dalex partial dependence
    if pd.api.types.is_numeric_dtype(X[feature]):
        explainer = dx.Explainer(model, X, y, label=model_name, verbose=False)
        profile = explainer.model_profile(variables=[feature], type="partial")
        res = profile.result.copy()

        if "_vname_" in res.columns:
            res = res[res["_vname_"] == feature]
        if "_label_" in res.columns:
            res = res[res["_label_"] == model_name]

        xcol = "_x_" if "_x_" in res.columns else feature
        ycol = "_yhat_" if "_yhat_" in res.columns else "yhat"
        res = res.sort_values(xcol)

        ax.plot(res[xcol].to_numpy(), res[ycol].to_numpy(), color=color)
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean prediction")
        ax.set_title(feature, fontweight="bold")

    # Categorical feature: manual intervention
    else:
        s = X[feature]
        levels = (
            list(s.cat.categories)
            if pd.api.types.is_categorical_dtype(s)
            else list(pd.Series(s).dropna().unique())
        )

        max_levels = 20
        levels = levels[:max_levels]

        means = []
        for lvl in levels:
            X_tmp = X.copy()
            X_tmp[feature] = lvl
            pred = model.predict(X_tmp)
            means.append(float(pd.Series(pred).mean()))

        out = pd.DataFrame({"level": levels, "mean_pred": means}).sort_values(
            "mean_pred", ascending=False
        )

        sns.barplot(
            data=out,
            x="level",
            y="mean_pred",
            ax=ax,
            palette=palette,
            color=color,
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean prediction")
        ax.set_title(feature, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

    if created_fig:
        fig.tight_layout()
        savefig(save_path, fig=fig)
        return fig

    return None

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_growth_rate_columns(
    df,
    columns,
    shift_period=-1,
    suffix="_growth_rate",
    as_percent=True,
    drop_original=False,
):
    '''
    Add growth rate columns to df based on specified columns.
    Note shift is negative because data is arranged from newest to oldest.
    '''
    for col in columns:
        base = pd.to_numeric(df[col], errors="coerce")
        growth = (base - base.shift(shift_period)) / base.shift(shift_period)
        if as_percent:
            growth = growth * 100

        new_col = f"{col}{suffix}"

        # Insert/move new column right beside the source column.
        if new_col in df.columns:
            df.pop(new_col)
        idx = df.columns.get_loc(col)
        df.insert(idx + 1, new_col, growth)

    if drop_original:
        drop_cols = [col for col in columns if col in df.columns]
        df.drop(columns=drop_cols, inplace=True)

    return df


def add_lag_columns(df, col_names, lags, keep_original=("quota_premium_growth_rate","quota_premium"), inplace=False):
    """
    Add lagged versions of columns to df.
    """
    out = df if inplace else df.copy()

    if isinstance(col_names, str):
        col_names = [col_names]

    if isinstance(lags, int):
        lag_list = [lags]
    elif isinstance(lags, range):
        lag_list = list(lags)
    elif isinstance(lags, tuple) and len(lags) == 2:
        start, end = lags
        lag_list = list(range(start, end + 1))  # inclusive
    else:
        lag_list = sorted(set(lags))

    lagged = {
        f"{col}_lag_{l}": out[col].shift(-l)
        for col in col_names
        for l in lag_list
    }
    if lagged:
        out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)

    drop_cols = [c for c in col_names if c not in set(keep_original)]
    out = out.drop(columns=drop_cols).copy()

    return out


def plot_quota_premium_vs_column(
    df,
    right_col,
    date_col="date",
    left_col="quota_premium",
    left_color="tab:blue",
    right_color="tab:orange",
    figsize=(10, 5),
    title=None,
    right_label=None,
):
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    l1, = ax_left.plot(df[date_col], df[left_col], color=left_color, label=left_col)
    ax_left.set_ylabel(left_col, color=left_color)
    ax_left.tick_params(axis="y", labelcolor=left_color)

    l2, = ax_right.plot(df[date_col], df[right_col], color=right_color, label=right_col)
    ax_right.set_ylabel(right_label if right_label is not None else right_col, color=right_color)
    ax_right.tick_params(axis="y", labelcolor=right_color)

    ax_left.set_xlabel(date_col)
    ax_left.set_title(title if title is not None else f"{left_col} vs {right_col}")

    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


def plot_scatter_against_quota_premium(
    df,
    x_col,
    y_col="quota_premium",
    figsize=(7, 5),
    title=None,
    add_trendline=False,
    color="tab:blue",
    alpha=0.7,
):
    plot_df = df[[x_col, y_col]].dropna()

    plt.figure(figsize=figsize)
    plt.scatter(plot_df[x_col], plot_df[y_col], alpha=alpha, color=color)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title if title is not None else f"{y_col} vs {x_col}")

    if add_trendline and len(plot_df) >= 2:
        coeff = pd.Series(plot_df[y_col]).astype(float)
        base = pd.Series(plot_df[x_col]).astype(float)
        m, b = np.polyfit(base, coeff, 1)
        x_line = pd.Series([base.min(), base.max()])
        y_line = m * x_line + b
        plt.plot(x_line, y_line, color="tab:red", linewidth=2, label="trend")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

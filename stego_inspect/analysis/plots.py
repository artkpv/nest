"""
Plotting functions for stego_inspect evaluation results.

Generates plots from DataFrames produced by eval_reader. The functions in
the second half of this module (`plot_heatmap`, `plot_timeline`,
`plot_family_comparison`, `plot_error_analysis`, `plot_per_number_accuracy`,
`plot_refusal_analysis_combined`) reproduce the legacy
`stego_counting_plots.py` visuals against the legacy column schema produced
by `counting_frame.to_counting_dataframe`.
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .model_registry import MODEL_INFO, get_model_info, model_sort_key, parse_model_name_and_version

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PUBLICATION_FORMATS: tuple[str, ...] = ("pdf",)


def _save_figure(
    fig, output_path, formats: tuple[str, ...] = PUBLICATION_FORMATS
) -> None:
    if output_path is None:
        return
    p = Path(output_path)
    for ext in formats:
        out = p.with_suffix(f".{ext}")
        kwargs: dict = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 150
        fig.savefig(out, **kwargs)


def _short_name(model_id: str) -> str:
    return get_model_info(model_id)["short_name"]


def _monitor_suffix(df: pd.DataFrame) -> str:
    if "monitor_model" not in df.columns:
        return ""
    monitors = sorted({m for m in df["monitor_model"].dropna().unique() if m})
    if not monitors:
        return ""
    names = ", ".join(_short_name(m) for m in monitors)
    return f" — monitor: {names}"


def _format_algorithm_label(name: str) -> str:
    """Display-friendly algorithm name.

    'sentence_acrostic_with_mapping' → 'sentence_acrostic (mapping)'.
    Bare algorithm names (mapping off or unsupported) are returned unchanged.
    """
    if not name:
        return name
    suffix = "_with_mapping"
    if name.endswith(suffix):
        return f"{name[: -len(suffix)]} (mapping)"
    return name


def _algorithm_suffix(df: pd.DataFrame, multiline: bool = True) -> str:
    """Return a suptitle suffix listing algorithms found in the frame."""
    algos = sorted(
        {a for a in df.get("algorithm", pd.Series(dtype=object)).dropna().unique() if a}
    )
    if not algos:
        return ""
    label = ", ".join(_format_algorithm_label(a) for a in algos)
    sep = "\n" if multiline else " — "
    return f"{sep}algorithm: {label}"


# Color palette
MOLTEN_LAVA = "#780000"
BRICK_RED = "#c1121f"
PAPAYA_WHIP = "#fdf0d5"
DEEP_SPACE_BLUE = "#003049"
STEEL_BLUE = "#669bbc"

PALETTE = [DEEP_SPACE_BLUE, BRICK_RED, STEEL_BLUE, MOLTEN_LAVA, PAPAYA_WHIP]


def plot_heatmap_generic(
    df: pd.DataFrame,
    metric: str = "hidden_task_scorer",
    row: str = "model",
    col: str = "task_arg__algorithm",
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".2f",
) -> plt.Figure:
    """
    Plot a heatmap of a metric by model x condition.

    Args:
        df: DataFrame from eval_reader
        metric: Column name for the metric to plot
        row: Column for heatmap rows (default: model)
        col: Column for heatmap columns (default: algorithm)
        title: Plot title
        output_path: If set, save figure to this path
        figsize: Figure size
        vmin/vmax: Color scale bounds
        fmt: Number format for annotations

    Returns:
        matplotlib Figure
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

    # Pivot to get row x col matrix of metric means
    pivot = df.pivot_table(values=metric, index=row, columns=col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric},
    )

    ax.set_title(title or f"{metric} by {row} x {col}")
    ax.set_ylabel(row)
    ax.set_xlabel(col)
    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_model_comparison(
    df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Bar chart comparing models on key metrics.

    Args:
        df: DataFrame from eval_reader
        metrics: List of metric columns to plot. Auto-detected if None.
        title: Plot title
        output_path: If set, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Auto-detect scorer columns
    if metrics is None:
        scorer_cols = [c for c in df.columns if c.endswith("_scorer") and "__" not in c]
        if not scorer_cols:
            raise ValueError("No scorer columns found. Provide metrics explicitly.")
        metrics = scorer_cols

    # Compute mean per model
    model_means = df.groupby("model")[metrics].mean()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(model_means))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            model_means[metric],
            width,
            label=metric.replace("_scorer", ""),
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_ylabel("Score")
    ax.set_title(title or "Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_means.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def generate_plots(
    df: pd.DataFrame,
    output_dir: str = "./plots",
) -> list[str]:
    """
    Generate standard set of plots from eval results.

    Args:
        df: DataFrame from load_all_results()
        output_dir: Directory to save plot images

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Detect available numeric scorer columns
    scorer_cols = [
        c
        for c in df.columns
        if c.endswith("_scorer")
        and "__" not in c
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]

    if not scorer_cols:
        print("No scorer columns found in data.")
        return saved

    # 1. Model comparison bar chart
    path = str(output_dir / "model_comparison.png")
    plot_model_comparison(df, metrics=scorer_cols, output_path=path)
    saved.append(path)
    plt.close()

    # 2. Heatmaps for each metric
    # Determine best column dimension
    col_candidates = ["task_arg__algorithm", "task_arg__hidden_dataset", "model"]
    col_dim = None
    for c in col_candidates:
        if c in df.columns and c != "model" and df[c].nunique() > 1:
            col_dim = c
            break

    if col_dim:
        for metric in scorer_cols:
            path = str(output_dir / f"heatmap_{metric}.png")
            plot_heatmap_generic(df, metric=metric, col=col_dim, output_path=path)
            saved.append(path)
            plt.close()

    print(f"Saved {len(saved)} plots to {output_dir}")
    return saved


# ---------------------------------------------------------------------------
# Stego-edge plot: condition comparison across difficulties, faceted by task.
# ---------------------------------------------------------------------------

CONDITION_COLORS = {
    "cot": DEEP_SPACE_BLUE,
    "stego": BRICK_RED,
    "direct": "#888888",
    "ttc": STEEL_BLUE,
    "dual": MOLTEN_LAVA,
}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def plot_condition_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    conditions: Optional[list[str]] = None,
    require_ceiling: bool = False,
    with_monitor_row: bool = False,
) -> plt.Figure:
    """Grid of grouped-bar plots: rows=model, cols=task_type.

    Expects the tidy frame from `condition_comparison.build_condition_frame`.
    The stego bar is split vertically: inner darker segment = stego_success
    rate (stego_correct AND NOT explicit_reasoning AND NOT direct AND NOT ttc);
    translucent outer bar reaches stego_correct rate. Diamond markers on the
    stego bar show stego_attempted rate over rows where the value isn't None.

    When `with_monitor_row=True`, each model row gets a thin sub-row beneath
    showing rate(monitor caught | real_stego) and rate(stego_detected |
    real_stego) per difficulty — i.e. how often the monitor flags genuine
    stego attempts.
    """
    from .condition_comparison import natural_sort_difficulties

    if conditions is None:
        conditions = ["cot", "stego", "direct", "ttc"]

    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if output_path:
            _save_figure(fig, output_path)
        return fig

    if require_ceiling:
        df = df[df["ceiling_ok"] == True]  # noqa: E712
        if df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No ceiling-passing samples", ha="center", va="center")
            ax.axis("off")
            if output_path:
                _save_figure(fig, output_path)
            return fig

    task_types = sorted(df["task_type"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    n_models = max(1, len(models))
    n_tasks = max(1, len(task_types))

    n_cols = n_tasks

    fig, axes = plt.subplots(
        n_models,
        n_cols,
        figsize=(max(5, 3.5 * n_tasks), max(3.5, 2.8 * n_models)),
        squeeze=False,
        sharey=True,
    )

    present_conditions = [c for c in conditions if c in df["condition"].unique()]
    # Split stego into two side-by-side bars: correct + unique success.
    display_conditions = []
    for c in present_conditions:
        display_conditions.append(c)
        if c == "stego":
            display_conditions.append("stego_success")
    width = 0.8 / max(1, len(display_conditions))

    stego_idx = (
        display_conditions.index("stego") if "stego" in display_conditions else None
    )

    for r, model in enumerate(models):
        for c, task_type in enumerate(task_types):
            ax = axes[r][c]
            ax.set_ylim(0, 1.05)
            sub = df[(df["task_type"] == task_type) & (df["model"] == model)]
            diffs = natural_sort_difficulties(sub["difficulty"].dropna().unique())
            if not diffs:
                diffs = [""]
            x = np.arange(len(diffs))

            for i, cond in enumerate(display_conditions):
                offset = (i - len(display_conditions) / 2 + 0.5) * width

                if cond == "stego_success":
                    color = CONDITION_COLORS.get("stego", PALETTE[i % len(PALETTE)])
                    success_means, success_errs = [], []
                    for d in diffs:
                        cell = sub[
                            (sub["condition"] == "stego") & (sub["difficulty"] == d)
                        ]
                        # stego_success is tri-state; None = monitor gave no verdict.
                        # Excluded, not counted as a failure — counting it would both
                        # deflate the rate and inflate the denominator (and hence
                        # shrink the SEM). See scorers/composite.py.
                        ss_vals = [
                            bool(v) for v in cell["stego_success"].tolist() if v is not None
                        ]
                        denom = len(ss_vals)
                        p = (sum(1 for v in ss_vals if v) / denom) if denom else 0.0
                        success_means.append(p)
                        se = (p * (1 - p) / denom) ** 0.5 if denom else 0.0
                        success_errs.append(se)
                    ax.bar(
                        x + offset,
                        success_means,
                        width,
                        color=color,
                        alpha=1.0,
                        edgecolor="black",
                        linewidth=0.8,
                        hatch="//",
                        yerr=success_errs,
                        capsize=2,
                        error_kw={"alpha": 0.8},
                        label="stego success" if (r == 0 and c == 0) else None,
                    )
                    for j, p in enumerate(success_means):
                        ax.annotate(
                            f"{p:.2f}",
                            (x[j] + offset, p + success_errs[j]),
                            textcoords="offset points",
                            xytext=(0, 3),
                            fontsize=6,
                            ha="center",
                            color="black",
                        )
                    continue

                means, lows, highs = [], [], []
                for d in diffs:
                    cell = sub[(sub["condition"] == cond) & (sub["difficulty"] == d)]
                    correct_vals = [
                        v for v in cell["correct"].tolist() if v is not None
                    ]
                    n = len(correct_vals)
                    k = sum(1 for v in correct_vals if v)
                    mean = k / n if n else 0.0
                    if n >= 5:
                        lo, hi = _wilson_ci(k, n)
                    else:
                        lo, hi = mean, mean
                    means.append(mean)
                    lows.append(max(0.0, mean - lo))
                    highs.append(max(0.0, hi - mean))
                color = CONDITION_COLORS.get(cond, PALETTE[i % len(PALETTE)])

                if cond == "stego":
                    ax.bar(
                        x + offset,
                        means,
                        width,
                        color=color,
                        alpha=0.85,
                        yerr=[lows, highs],
                        capsize=2,
                        error_kw={"alpha": 0.6},
                        label="failed stego" if (r == 0 and c == 0) else None,
                    )
                    # Diamond: caught rate (explicit_reasoning OR stego_detected).
                    for j, d in enumerate(diffs):
                        cell = sub[
                            (sub["condition"] == "stego") & (sub["difficulty"] == d)
                        ]
                        caught_vals = [v for v in _caught_list(cell) if v is not None]
                        if not caught_vals:
                            continue
                        caught_rate = sum(1 for v in caught_vals if v) / len(caught_vals)
                        ax.scatter(
                            x[j] + offset,
                            caught_rate,
                            marker="D",
                            s=30,
                            color="#e07a5f",
                            zorder=5,
                            label="caught (er ∨ sd)"
                            if (r == 0 and c == 0 and j == 0)
                            else None,
                        )
                else:
                    ax.bar(
                        x + offset,
                        means,
                        width,
                        color=color,
                        yerr=[lows, highs],
                        capsize=2,
                        error_kw={"alpha": 0.6},
                        label=cond if (r == 0 and c == 0) else None,
                    )

            # Per-difficulty sample count (max N across conditions for that diff).
            tick_labels = []
            for d in diffs:
                cell_d = sub[sub["difficulty"] == d]
                n_per_cond = cell_d.groupby("condition")["correct"].apply(
                    lambda s: s.notna().sum()
                )
                n_max = int(n_per_cond.max()) if len(n_per_cond) else 0
                label = f"{d}\nn={n_max}" if d else f"n={n_max}"
                tick_labels.append(label)
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)
            if r == 0:
                ax.set_title(task_type, fontsize=10)
            if c == 0:
                ax.set_ylabel(_short_name(model), fontsize=8)

    algo_suffix = _algorithm_suffix(df)
    fig_h = fig.get_figheight()
    fig.suptitle(
        "Stego Edge: Condition Accuracy by Model × Task"
        + (" (ceiling-filtered)" if require_ceiling else "")
        + algo_suffix
        + _monitor_suffix(df),
        y=1 - 0.05 / fig_h,
        va="top",
    )
    handles, labels = [], []
    seen = set()
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l and l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(labels), 6),
        fontsize=8,
        frameon=True,
    )
    # Reserve fixed inches for the legend and the two-line title (anchored a
    # fixed distance from the top): fractions of the height leave ~1 in of
    # blank space on a tall many-model grid.
    fig.tight_layout(rect=[0, 0.4 / fig_h, 1.0, 1 - 0.5 / fig_h])
    # tight_layout over-reserves for the suptitle; pin top and bottom in inches
    # (title 0.5 in + column titles 0.25 in; legend 0.4 in + tick labels 0.35 in).
    fig.subplots_adjust(top=1 - 0.75 / fig_h, bottom=0.75 / fig_h)

    if output_path:
        _save_figure(fig, output_path)
    return fig


MIN_STEGO_FOR_PLOT = 5


def _stego_rows(df: pd.DataFrame) -> pd.DataFrame:
    """All stego-condition rows."""
    if df.empty:
        return df
    return df[df["condition"] == "stego"]


def _rate_over(values: list, predicate=lambda v: v is True) -> tuple[float | None, int]:
    obs = [v for v in values if v is not None]
    n = len(obs)
    if n == 0:
        return None, 0
    return sum(1 for v in obs if predicate(v)) / n, n


def _caught_value(er, sd) -> bool | None:
    """OR-union of explicit_reasoning and stego_detected; tri-state."""
    if er is True or sd is True:
        return True
    if er is None and sd is None:
        return None
    return False


def _caught_list(rows: pd.DataFrame) -> list[bool | None]:
    return [
        _caught_value(er, sd)
        for er, sd in zip(rows["explicit_reasoning"].tolist(), rows["stego_detected"].tolist())
    ]


def plot_stego_stealth(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """2D Pareto: stego_success × stealth, faceted by family.

    Samples where explicit_reasoning=True are excluded first (they already
    failed stealth via plain-text leakage). Over the remaining samples:
    X = rate(stego_success), Y = 1 − rate(stego_detected).

    Top-right is the win zone (covert encoding AND solves the task).
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if output_path:
            _save_figure(fig, output_path)
        return fig

    stego = _stego_rows(df)
    task_types = sorted(df["task_type"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    n_panels = max(1, len(task_types))
    n_cols = min(2, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 4.8 * n_cols), 4.4 * n_rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    # Hide any unused panels
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    model_color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    for c, t in enumerate(task_types):
        ax = axes_flat[c]
        sub = stego[stego["task_type"] == t]
        for m in models:
            cell = sub[sub["model"] == m]
            # Exclude samples where the monitor already saw explicit reasoning
            cell = cell[cell["explicit_reasoning"] != True]  # noqa: E712
            # Tri-state: drop samples with no monitor verdict rather than scoring
            # them as failures (see scorers/composite.py). `n` must be the scored
            # denominator, not the row count, or the plotted rate and its opacity
            # threshold are both computed against undetermined samples.
            ss_vals = [
                bool(v) for v in cell["stego_success"].tolist() if v is not None
            ]
            n = len(ss_vals)
            if n == 0:
                continue
            ss_k = sum(ss_vals)
            success_rate = ss_k / n
            # Y-axis: stego_detected only (explicit_reasoning already filtered out)
            sd_rate, _ = _rate_over(cell["stego_detected"].tolist())
            if sd_rate is None:
                continue
            stealth = 1.0 - sd_rate
            color = model_color[m]
            alpha = 0.95 if n >= MIN_STEGO_FOR_PLOT else 0.35

            def _binom_sem(p: float, n: int) -> float:
                return (p * (1 - p) / n) ** 0.5 if n > 0 else 0.0

            xerr = _binom_sem(success_rate, n)
            yerr = _binom_sem(sd_rate, n)
            ax.errorbar(
                success_rate,
                stealth,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=1.4,
                capsize=4,
                capthick=1.2,
                alpha=alpha,
                zorder=3,
            )
            ax.plot(
                success_rate,
                stealth,
                marker="o",
                markersize=4,
                color=color,
                alpha=alpha,
                label=_short_name(m) if c == 0 else None,
                linestyle="none",
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=4,
            )
            ax.annotate(
                f"n={n}",
                (success_rate, stealth),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                color=color,
                alpha=0.85,
            )

        ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.axvline(0.5, color="gray", linewidth=0.4, linestyle=":", alpha=0.3)
        ax.axhline(0.5, color="gray", linewidth=0.4, linestyle=":", alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("P(stego_success)", fontsize=9)
        if c % n_cols == 0:
            ax.set_ylabel("stealth = 1 − P(stego_detected)", fontsize=9)
        ax.grid(True, alpha=0.25)

    algo_suffix = _algorithm_suffix(df)
    fig.suptitle(
        "Stego stealth × success rate (excl. explicit-reasoning samples)"
        + algo_suffix
        + _monitor_suffix(df)
    )

    handles, labels = [], []
    seen = set()
    for ax in axes_flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(len(labels), 6),
            fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0.05, 1.0, 0.95])

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_stego_outcome_mix(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """100% stacked bars per (model, family) over real-stego samples.

    Segments (bottom→top): stealth-win (✓ uncaught), leaky stego (✓ caught),
    wasted stealth (✗ uncaught), full failure (✗ caught). The bottom green
    segment is the headline number — fraction of real-stego attempts that both
    solved the task and slipped past the monitor.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if output_path:
            _save_figure(fig, output_path)
        return fig

    stego = _stego_rows(df)
    task_types = sorted(df["task_type"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    n_cols = max(1, len(task_types))

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(max(5, 1.0 * len(models) * n_cols + 2), 5.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    seg_colors = {
        "win": "#3a7d44",       # stealth win — green
        "leaky": "#e9c46a",     # correct but caught — yellow
        "wasted": "#cccccc",    # uncaught but wrong — light grey
        "fail": "#5a5a5a",      # caught and wrong — dark grey
    }

    for c, t in enumerate(task_types):
        ax = axes[c]
        sub = stego[stego["task_type"] == t]
        x = np.arange(len(models))
        wins, leakys, wasteds, fails, ns = [], [], [], [], []
        for m in models:
            cell = sub[sub["model"] == m]
            caught_vals = _caught_list(cell)
            counted = [
                (bool(corr), bool(caught))
                for corr, caught in zip(cell["correct"].tolist(), caught_vals)
                if corr is not None and caught is not None
            ]
            n = len(counted)
            ns.append(n)
            if n == 0:
                wins.append(0); leakys.append(0); wasteds.append(0); fails.append(0)
                continue
            w = sum(1 for corr, caught in counted if corr and not caught)
            l_ = sum(1 for corr, caught in counted if corr and caught)
            ws = sum(1 for corr, caught in counted if not corr and not caught)
            f = sum(1 for corr, caught in counted if not corr and caught)
            wins.append(w / n)
            leakys.append(l_ / n)
            wasteds.append(ws / n)
            fails.append(f / n)

        wins = np.asarray(wins); leakys = np.asarray(leakys)
        wasteds = np.asarray(wasteds); fails = np.asarray(fails)
        ax.bar(x, wins, color=seg_colors["win"], label="stealth win (✓ uncaught)" if c == 0 else None, edgecolor="black", linewidth=0.4)
        ax.bar(x, leakys, bottom=wins, color=seg_colors["leaky"], label="leaky stego (✓ caught)" if c == 0 else None, edgecolor="black", linewidth=0.4)
        ax.bar(x, wasteds, bottom=wins + leakys, color=seg_colors["wasted"], label="wasted stealth (✗ uncaught)" if c == 0 else None, edgecolor="black", linewidth=0.4)
        ax.bar(x, fails, bottom=wins + leakys + wasteds, color=seg_colors["fail"], label="full failure (✗ caught)" if c == 0 else None, edgecolor="black", linewidth=0.4)

        # Annotate stealth-win % at top of green segment
        for xi, w, n in zip(x, wins, ns):
            if n > 0:
                ax.text(xi, w / 2, f"{w * 100:.0f}", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
                ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom", fontsize=6, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([_short_name(m) for m in models], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.10)
        ax.set_title(t, fontsize=10)
        if c == 0:
            ax.set_ylabel("share of real-stego samples", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    algo_suffix = _algorithm_suffix(df)
    fig.suptitle(
        "Stego outcome mix (all stego samples; caught = stego_detected ∨ explicit_reasoning)"
        + algo_suffix
        + _monitor_suffix(df)
    )

    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(len(labels), 4),
            fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0.06, 1.0, 0.94])

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_monitor_confidence(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Per (model, family), split distribution of monitor_confidence.

    Compares monitor_confidence between stego samples where the model solved
    the hidden task (stego_correct=True) vs samples where it didn't.
    Diagnostic: does the monitor's continuous score shift higher when the
    model actually does the hidden task — i.e. is the monitor sensitive to
    the cases we most want it to catch?
    """
    if df.empty or "monitor_confidence" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if output_path:
            _save_figure(fig, output_path)
        return fig

    stego = df[df["condition"] == "stego"].copy()
    stego["monitor_confidence"] = pd.to_numeric(
        stego["monitor_confidence"], errors="coerce"
    )
    stego = stego.dropna(subset=["monitor_confidence"])
    if stego.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No monitor_confidence values", ha="center", va="center")
        ax.axis("off")
        if output_path:
            _save_figure(fig, output_path)
        return fig

    task_types = sorted(stego["task_type"].dropna().unique())
    models = sorted(stego["model"].dropna().unique())
    n_rows = max(1, len(models))
    n_cols = max(1, len(task_types))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5, 2.8 * n_cols), max(3.0, 1.8 * n_rows)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    correct_color = BRICK_RED
    incorrect_color = STEEL_BLUE

    for r, m in enumerate(models):
        for c, t in enumerate(task_types):
            ax = axes[r][c]
            cell = stego[(stego["model"] == m) & (stego["task_type"] == t)]
            correct_vals = cell.loc[
                cell["correct"] == True,  # noqa: E712
                "monitor_confidence",
            ].tolist()
            incorrect_vals = cell.loc[
                cell["correct"] == False,  # noqa: E712
                "monitor_confidence",
            ].tolist()

            if correct_vals:
                ax.hist(
                    correct_vals,
                    bins=np.linspace(0, 1, 11),
                    alpha=0.55,
                    color=correct_color,
                    label="stego_correct=True" if (r == 0 and c == 0) else None,
                    edgecolor="black",
                    linewidth=0.4,
                    density=True,
                )
            if incorrect_vals:
                ax.hist(
                    incorrect_vals,
                    bins=np.linspace(0, 1, 11),
                    alpha=0.55,
                    color=incorrect_color,
                    label="stego_correct=False" if (r == 0 and c == 0) else None,
                    edgecolor="black",
                    linewidth=0.4,
                    density=True,
                )
            ax.text(
                0.98,
                0.92,
                f"n={len(correct_vals)}/{len(incorrect_vals)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                color="black",
            )
            ax.set_xlim(-0.02, 1.02)
            if r == 0:
                ax.set_title(t, fontsize=9)
            if c == 0:
                ax.set_ylabel(_short_name(m), fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("monitor_confidence", fontsize=8)
            ax.grid(axis="y", alpha=0.2)

    algo_suffix = _algorithm_suffix(df)
    fig.suptitle(
        "Monitor confidence: stego_correct=True vs False (all stego samples)"
        + algo_suffix
        + _monitor_suffix(df),
        fontsize=11,
    )

    handles, labels = [], []
    seen = set()
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l and l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(labels), 4),
            fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0.05, 1.0, 0.95])

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_stego_uplift(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bars of stego-success rate per model × task_type.

    Each task_type gets a group of bars, one per model. A dashed gray line
    shows the CoT-baseline accuracy (averaged across models) as a capability
    ceiling.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if output_path:
            _save_figure(fig, output_path)
        return fig

    task_types = sorted(df["task_type"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    n_models = len(models)
    n_tasks = len(task_types)

    def _binom_sem(k: int, n: int) -> float:
        if n <= 0:
            return 0.0
        p = k / n
        return (p * (1 - p) / n) ** 0.5

    fig, axes = plt.subplots(
        1,
        n_tasks,
        sharey=True,
        figsize=(max(4, 1.0 * n_models) * n_tasks + 1.5, 5),
        gridspec_kw={"wspace": 0.08},
    )
    if n_tasks == 1:
        axes = [axes]

    x = np.arange(n_models)
    cot_handle = None

    # One marker style per monitor model (so multiple monitors → multiple lines).
    monitors: list[str] = []
    if "monitor_model" in df.columns:
        monitors = sorted({m for m in df["monitor_model"].dropna().unique() if m})
    monitor_markers = ["o", "s", "D", "^", "v", "P", "X"]
    monitor_color = "#222222"
    monitor_handles: dict[str, object] = {}

    for t_idx, (ax, t) in enumerate(zip(axes, task_types)):
        ys, errs, ns = [], [], []
        for model in models:
            cell = df[
                (df["model"] == model)
                & (df["task_type"] == t)
                & (df["condition"] == "stego")
            ]
            # Tri-state: undetermined (None) samples are excluded from the
            # denominator, not scored as failures. See scorers/composite.py.
            ss = [bool(v) for v in cell["stego_success"].tolist() if v is not None]
            n = len(ss)
            k = sum(1 for v in ss if v)
            ys.append((k / n) if n else 0.0)
            errs.append(_binom_sem(k, n) if n else 0.0)
            ns.append(n)

        bar_colors = [PALETTE[i % len(PALETTE)] for i in range(n_models)]
        bars = ax.bar(
            x,
            ys,
            0.75,
            yerr=errs,
            color=bar_colors,
            capsize=2,
            error_kw={"alpha": 0.7, "elinewidth": 1},
            alpha=0.85,
        )

        # Value labels above each bar (above the error cap).
        for xi, (yi, ei, ni) in enumerate(zip(ys, errs, ns)):
            if ni == 0:
                ax.text(xi, 0.02, "n=0", ha="center", va="bottom", fontsize=7, color="gray")
                continue
            ax.text(
                xi,
                min(yi + ei + 0.02, 1.02),
                f"{yi:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

        # CoT ceiling: dashed line + ±SEM band across the subplot.
        cot_cell = df[(df["task_type"] == t) & (df["condition"] == "cot")]
        cot_vals = [v for v in cot_cell["correct"].tolist() if v is not None]
        cot_n = len(cot_vals)
        if cot_n:
            cot_k = sum(1 for v in cot_vals if v)
            cot_y = cot_k / cot_n
            cot_se = _binom_sem(cot_k, cot_n)
            xmin, xmax = -0.6, n_models - 0.4
            ax.fill_between(
                [xmin, xmax],
                [cot_y - cot_se] * 2,
                [cot_y + cot_se] * 2,
                color="gray",
                alpha=0.15,
                zorder=4,
            )
            line = ax.hlines(
                cot_y, xmin, xmax,
                colors="gray", linewidths=1.8, linestyles="dashed", alpha=0.85, zorder=5,
            )
            if cot_handle is None:
                cot_handle = line
            ax.text(
                xmax, cot_y, f" CoT={cot_y:.2f}",
                va="center", ha="left", fontsize=7, color="gray", zorder=6,
            )

        # Overlay: monitor stego_detected rate per (model) — one line per monitor.
        for mi, mon in enumerate(monitors):
            xs_m, ys_m, errs_m = [], [], []
            for x_i, model in enumerate(models):
                cell = df[
                    (df["model"] == model)
                    & (df["task_type"] == t)
                    & (df["condition"] == "stego")
                    & (df["monitor_model"] == mon)
                ]
                vals = [v for v in cell["stego_detected"].tolist() if v is not None]
                if not vals:
                    continue
                k_m = sum(1 for v in vals if v)
                n_m = len(vals)
                xs_m.append(x_i)
                ys_m.append(k_m / n_m)
                errs_m.append(_binom_sem(k_m, n_m))
            if not xs_m:
                continue
            marker = monitor_markers[mi % len(monitor_markers)]
            line, = ax.plot(
                xs_m, ys_m,
                color=monitor_color, marker=marker, markersize=6,
                linestyle="-", linewidth=1.2, alpha=0.9, zorder=6,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"monitor: {_short_name(mon)} (P(stego_detected))",
            )
            ax.errorbar(
                xs_m, ys_m, yerr=errs_m,
                fmt="none", ecolor=monitor_color, elinewidth=0.8,
                capsize=2, alpha=0.6, zorder=5,
            )
            monitor_handles.setdefault(mon, line)

        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_models)
        ax.set_xlim(-0.6, n_models - 0.4)
        ax.set_title(f"{t}\n(stego n={sum(ns)}, cot n={cot_n})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if t_idx == 0:
            ax.set_ylabel("Stego success rate")

    axes[0].set_ylim(0, 1.08)

    # Right-side label: what the scatter line means.
    if monitors:
        ax_right = axes[-1].twinx()
        ax_right.set_ylim(axes[0].get_ylim())
        ax_right.set_yticks([])
        ax_right.set_ylabel(
            "Monitor stego_detected rate\n(scatter line; per (model, family) cell)",
            fontsize=9, color=monitor_color, rotation=270, labelpad=18,
        )

    # Build legend: one swatch per model + the CoT line.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=PALETTE[i % len(PALETTE)], alpha=0.85, label=_short_name(m))
        for i, m in enumerate(models)
    ]
    handles.append(Line2D([0], [0], color="gray", linestyle="dashed", linewidth=1.8, label="CoT baseline (±SEM)"))
    for mi, mon in enumerate(monitors):
        handles.append(Line2D(
            [0], [0],
            color=monitor_color,
            marker=monitor_markers[mi % len(monitor_markers)],
            markersize=6, markeredgecolor="white", markeredgewidth=0.6,
            linestyle="-", linewidth=1.2,
            label=f"monitor: {_short_name(mon)} (P(stego_detected))",
        ))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=8, frameon=False)

    algo_suffix = _algorithm_suffix(df)
    fig.suptitle("Stego CoT Uplift: success when baselines fail" + algo_suffix, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    if output_path:
        _save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Legacy stego_counting plots, ported to operate on a counting-shaped DataFrame
# (see counting_frame.to_counting_dataframe). Each plot expects the legacy
# columns: short_name, family, release_date, condition, exact_match,
# per_number_accuracy, longest_match, refused, sentence_count_expected,
# sentence_count_actual, D.
# ---------------------------------------------------------------------------


def _counting_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "stego", [MOLTEN_LAVA, BRICK_RED, PAPAYA_WHIP, STEEL_BLUE, DEEP_SPACE_BLUE]
    )


def _format_pct_cbar(ax):
    from matplotlib.ticker import FuncFormatter

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))


def _model_order_by_family(short_names) -> list[str]:
    """Order models by (family, release_date) using MODEL_INFO."""
    ordered = []
    seen = set(short_names)
    for _, info in sorted(
        MODEL_INFO.items(), key=lambda x: (x[1]["family"], x[1]["release_date"])
    ):
        if info["short_name"] in seen and info["short_name"] not in ordered:
            ordered.append(info["short_name"])
    # Append any unknown models at the end.
    for name in short_names:
        if name not in ordered:
            ordered.append(name)
    return ordered


# Models whose samples were partly excluded because generation was truncated
# (set by the counting CLI under --exclude-truncated). Their tick labels get a
# dagger and their timeline bars are hatched, so readers see the rates are
# survivor-only and biased upward. `*` is avoided: it marks significance.
TRUNCATION_MARKED: set[str] = set()
# (short_name, D) of each excluded truncated sample, so the timeline can count
# provider-side refusals alongside verbalized ones.
TRUNCATED_SAMPLES: list[tuple[str, int]] = []


# Order model rows/bars by provider, family, version, then name
# (`model_sort_key`), so related models sit together. False = each plot's
# own legacy order (the default).
PROVIDER_ORDER = False


def _ordered(order: list[str]) -> list[str]:
    return sorted(order, key=model_sort_key) if PROVIDER_ORDER else order


def _dagger(label: str) -> str:
    return f"{label}\u2020" if label in TRUNCATION_MARKED else label


def _mark_truncated_yticks(ax) -> None:
    if TRUNCATION_MARKED:
        ax.set_yticklabels([_dagger(t.get_text()) for t in ax.get_yticklabels()])


def plot_heatmap(
    df: pd.DataFrame,
    output_path: Path | None = None,
    condition_order: list[str] | None = None,
    condition_labels: dict[str, str] | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot 1: Heatmap of `exact_match` rate by model × condition.

    Annotates each cell with `pct(sem) n`, ordered by model family. Mirrors
    legacy `stego_counting_plots.plot_heatmap`.
    """
    if condition_order is None:
        condition_order = sorted(df["condition"].unique())
    if condition_labels is None:
        condition_labels = {c: c for c in condition_order}

    agg = (
        df.groupby(["short_name", "condition"])
        .agg(
            exact_match=("exact_match", "mean"),
            sem=("exact_match", "sem"),
            count=("exact_match", "count"),
        )
        .reset_index()
    )

    pivot_mean = agg.pivot(
        index="short_name", columns="condition", values="exact_match"
    )
    pivot_sem = agg.pivot(index="short_name", columns="condition", values="sem")
    pivot_count = agg.pivot(index="short_name", columns="condition", values="count")

    cols = [c for c in condition_order if c in pivot_mean.columns]
    pivot_mean = pivot_mean[cols]
    pivot_sem = pivot_sem[cols]
    pivot_count = pivot_count[cols]

    model_order = _ordered(_model_order_by_family(pivot_mean.index.tolist()))
    pivot_mean = pivot_mean.reindex(model_order)
    pivot_sem = pivot_sem.reindex(model_order)
    pivot_count = pivot_count.reindex(model_order)

    annot = pivot_mean.copy().astype(object)
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            val = pivot_mean.iloc[i, j]
            sem = pivot_sem.iloc[i, j]
            count = pivot_count.iloc[i, j]
            if pd.notna(val) and pd.notna(count):
                pct = val * 100
                sem_pct = sem * 100 if pd.notna(sem) else 0
                annot.iloc[i, j] = f"{pct:.0f}({sem_pct:.1f}) {int(count)}"
            else:
                annot.iloc[i, j] = ""

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot_mean,
        annot=annot,
        fmt="",
        cmap=_counting_cmap(),
        vmin=0,
        vmax=1,
        ax=ax,
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Exact Match Rate (%)"},
    )
    _format_pct_cbar(ax)

    ax.set_xlabel("Condition")
    ax.set_ylabel("Model")
    ax.set_title(
        "Steganographic Counting: Exact Match Rate by Model and Condition"
        + title_suffix
    )
    ax.set_xticklabels(
        [condition_labels.get(c, c) for c in pivot_mean.columns],
        rotation=45,
        ha="right",
    )
    _mark_truncated_yticks(ax)
    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_per_number_accuracy(
    df: pd.DataFrame,
    output_path: Path | None = None,
    condition_order: list[str] | None = None,
    condition_labels: dict[str, str] | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot 5: Per-number accuracy heatmap (partial credit). Mirrors legacy
    `stego_counting_plots.plot_per_number_accuracy`.
    """
    if condition_order is None:
        condition_order = sorted(df["condition"].unique())
    if condition_labels is None:
        condition_labels = {c: c for c in condition_order}

    agg = (
        df.groupby(["short_name", "condition"])
        .agg(
            mean=("per_number_accuracy", "mean"),
            sem=("per_number_accuracy", "sem"),
            count=("per_number_accuracy", "count"),
        )
        .reset_index()
    )

    pivot_mean = agg.pivot(index="short_name", columns="condition", values="mean")
    pivot_sem = agg.pivot(index="short_name", columns="condition", values="sem")
    pivot_count = agg.pivot(index="short_name", columns="condition", values="count")

    cols = [c for c in condition_order if c in pivot_mean.columns]
    pivot_mean = pivot_mean[cols]
    pivot_sem = pivot_sem[cols]
    pivot_count = pivot_count[cols]

    # Sort by company, model name, version (descending) like legacy.
    model_sort = []
    for short in pivot_mean.index:
        company, name, version = parse_model_name_and_version(short)
        model_sort.append((short, (company, name, -version)))
    model_order = _ordered([s for s, _ in sorted(model_sort, key=lambda x: x[1])])

    pivot_mean = pivot_mean.reindex(model_order)
    pivot_sem = pivot_sem.reindex(model_order)
    pivot_count = pivot_count.reindex(model_order)

    annot = pivot_mean.copy().astype(object)
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            val = pivot_mean.iloc[i, j]
            sem = pivot_sem.iloc[i, j]
            count = pivot_count.iloc[i, j]
            if pd.notna(val) and pd.notna(count):
                pct = val * 100
                sem_pct = sem * 100 if pd.notna(sem) else 0
                annot.iloc[i, j] = f"{pct:.0f}({sem_pct:.1f}) {int(count)}"
            else:
                annot.iloc[i, j] = ""

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot_mean,
        annot=annot,
        fmt="",
        cmap=_counting_cmap(),
        vmin=0,
        vmax=1,
        ax=ax,
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Per-Number Accuracy (%)"},
    )
    _format_pct_cbar(ax)

    ax.set_xlabel("Condition")
    ax.set_ylabel("Model")
    ax.set_title(
        "Steganographic Counting: Per-Number Accuracy (Partial Credit)" + title_suffix
    )
    ax.set_xticklabels(
        [condition_labels.get(c, c) for c in pivot_mean.columns],
        rotation=45,
        ha="right",
    )
    _mark_truncated_yticks(ax)
    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_timeline(
    df: pd.DataFrame,
    output_path: Path | None = None,
    title_suffix: str = "",
    bar_stats: pd.DataFrame | None = None,
    monitor_df: pd.DataFrame | None = None,
) -> Optional[plt.Figure]:
    """
    Plot 2: Per-model bar chart of per-number accuracy with company/model
    group separators and refusal annotations. Mirrors legacy
    `stego_counting_plots.plot_timeline`. Refusal counts come from the
    `refused` column in the DataFrame instead of raw trial dicts.

    `bar_stats` (short_name, accuracy, ci95, n_samples) replaces the pooled
    per-model bars; `monitor_df` is the frame the monitor markers come from
    (default: `df`).
    """
    refusal_counts: dict[str, int] = defaultdict(int)
    for short, group in df.groupby("short_name"):
        refusal_counts[short] = int(group["refused"].sum())
    # Provider-side refusals (truncations) were dropped from df; add those at
    # the D values this plot covers so both kinds of refusal show as `ref`.
    d_values = set(df["D"].dropna().astype(int)) if "D" in df.columns else set()
    for short, d in TRUNCATED_SAMPLES:
        if d in d_values:
            refusal_counts[short] += 1

    agg = (
        df.groupby("short_name")
        .agg(
            accuracy=("per_number_accuracy", "mean"),
            std=("per_number_accuracy", "std"),
            n_samples=("per_number_accuracy", "count"),
        )
        .reset_index()
    )
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["n_samples"])
    if bar_stats is not None:
        agg = bar_stats
    if monitor_df is None:
        monitor_df = df

    rows = []
    for _, row in agg.iterrows():
        company, name, version = parse_model_name_and_version(row["short_name"])
        if company not in ("Anthropic", "OpenAI"):
            continue
        short = row["short_name"]
        rows.append(
            {
                "short_name": short,
                "accuracy": row["accuracy"],
                "ci95": row["ci95"],
                "n_samples": int(row["n_samples"]),
                "n_refusals": refusal_counts.get(short, 0),
                "company": company,
                "model_name": name,
                "version": version,
            }
        )
    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        print("    (skipped plot_timeline - no Anthropic/OpenAI models in data)")
        return None

    plot_df = plot_df.sort_values(["company", "model_name", "version"]).reset_index(
        drop=True
    )

    companies = plot_df["company"].unique()
    company_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(companies)}

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(plot_df))
    bar_colors = [company_colors[c] for c in plot_df["company"]]

    bars = ax.bar(
        x,
        plot_df["accuracy"],
        width=0.7,
        yerr=plot_df["ci95"].fillna(0),
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )

    for bar, n, ci, acc, n_ref in zip(
        bars,
        plot_df["n_samples"],
        plot_df["ci95"].fillna(0),
        plot_df["accuracy"],
        plot_df["n_refusals"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{acc * 100:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        if n_ref > 0:
            ax.annotate(
                f"ref={n_ref}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci),
                xytext=(
                    bar.get_x() + bar.get_width() / 2 + 0.3,
                    # Tall bars: put the label beside the bar, below the n= text.
                    bar.get_height() + ci + 0.15
                    if bar.get_height() + ci + 0.15 <= 0.95
                    else bar.get_height() - 0.12,
                ),
                fontsize=7,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=BRICK_RED,
                    edgecolor="black",
                    linewidth=0.5,
                ),
                arrowprops=dict(arrowstyle="-", color=BRICK_RED, lw=1),
            )

    # Group separators (company / model name).
    prev_key = None
    model_group_starts = []
    for i, (company, mname) in enumerate(
        zip(plot_df["company"], plot_df["model_name"])
    ):
        key = (company, mname)
        if key != prev_key:
            if prev_key is not None:
                ax.axvline(
                    x=i - 0.5, color="gray", linewidth=0.6, linestyle=":", alpha=0.4
                )
            model_group_starts.append((i, company, mname))
            prev_key = key

    prev_company = None
    company_starts = []
    for i, company in enumerate(plot_df["company"]):
        if company != prev_company:
            if prev_company is not None:
                ax.axvline(
                    x=i - 0.5, color="black", linewidth=1.2, linestyle="-", alpha=0.6
                )
            company_starts.append((i, company))
            prev_company = company

    for idx, (start, company) in enumerate(company_starts):
        end = (
            company_starts[idx + 1][0]
            if idx + 1 < len(company_starts)
            else len(plot_df)
        )
        mid = (start + end - 1) / 2
        ax.text(
            mid,
            1.12,
            company,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    for idx, (start, company, mname) in enumerate(model_group_starts):
        end = (
            model_group_starts[idx + 1][0]
            if idx + 1 < len(model_group_starts)
            else len(plot_df)
        )
        mid = (start + end - 1) / 2
        ax.text(
            mid,
            1.05,
            mname,
            ha="center",
            va="bottom",
            fontsize=8,
            fontstyle="italic",
            color="gray",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    # Overlay: monitor stego_detected rate per model (scatter line). Mirrors
    # the monitor overlay on plot_stego_uplift for the RG sweep — counting
    # cells get the same diagnostic so accuracy and detection are visible
    # on the same figure.
    monitor_handle = None
    if "stego_detected" in monitor_df.columns:
        xs_m, ys_m, errs_m = [], [], []
        for xi, short in enumerate(plot_df["short_name"]):
            sub = monitor_df[monitor_df["short_name"] == short]
            # Optional `monitor_weight` re-weights stratified monitor samples to
            # the cell population; absent, every graded sample counts once.
            w_col = sub["monitor_weight"] if "monitor_weight" in sub.columns else None
            vals, ws = [], []
            for i, v in sub["stego_detected"].items():
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                vals.append(bool(v))
                w = 1.0 if w_col is None or pd.isna(w_col[i]) else float(w_col[i])
                ws.append(w)
            if not vals:
                continue
            p = sum(w for v, w in zip(vals, ws) if v) / sum(ws)
            n_m = sum(ws) ** 2 / sum(w * w for w in ws)  # Kish effective n
            xs_m.append(xi)
            ys_m.append(p)
            errs_m.append((p * (1 - p) / n_m) ** 0.5)
        if xs_m:
            line, = ax.plot(
                xs_m, ys_m,
                color="#fdbf2d", marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor="#5a3a00",
                markeredgewidth=1.2,
                linestyle="-", linewidth=2.0, alpha=1.0, zorder=7,
                path_effects=[
                    path_effects.withStroke(linewidth=4.0, foreground="black", alpha=0.7)
                ],
            )
            ax.errorbar(
                xs_m, ys_m, yerr=errs_m,
                fmt="none", ecolor="#5a3a00", elinewidth=1.0,
                capsize=2.5, alpha=0.8, zorder=6,
            )
            monitor_handle = line

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_dagger(n) for n in plot_df["short_name"]], rotation=45, ha="right", fontsize=8
    )
    for bar, name in zip(bars, plot_df["short_name"]):
        if name in TRUNCATION_MARKED:
            bar.set_hatch("///")
            bar.set_edgecolor("white")
    ax.set_ylabel("Rate")
    title = "Steganographic Counting: Per-Number Accuracy by Model" + title_suffix
    if monitor_handle is not None:
        title += _monitor_suffix(monitor_df)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(facecolor=company_colors[c], edgecolor="black", label=c)
        for c in companies
    ]
    legend_handles.append(Patch(facecolor="none", edgecolor="none", label="n = trials"))
    legend_handles.append(
        Patch(facecolor="none", edgecolor="none", label="ref = refusals")
    )
    if monitor_handle is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="#fdbf2d", marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor="#5a3a00",
                markeredgewidth=1.2,
                linestyle="-", linewidth=2.0,
                label="P(stego_detected) (monitor)",
            )
        )
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_family_comparison(
    df: pd.DataFrame,
    output_path: Path | None = None,
    condition_order: list[str] | None = None,
    condition_labels: dict[str, str] | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot 3: Grouped bar chart comparing per-number accuracy by model family
    across D conditions. Mirrors legacy `stego_counting_plots.plot_family_comparison`.
    """
    KEEP_CONDITIONS = {"D=4", "D=6", "D=8", "D=10", "D=12", "D=16", "D=26", "D=50"}
    KEEP_FAMILIES = {"Claude", "DeepSeek", "Grok", "GPT", "Qwen"}

    refusal_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in df.iterrows():
        family = row["family"]
        if family not in KEEP_FAMILIES:
            continue
        if row["refused"]:
            refusal_counts[(family, row["condition"])] += 1

    df = df[df["family"].isin(KEEP_FAMILIES)].copy()

    if condition_order is None:
        condition_order = sorted(df["condition"].unique())
    if condition_labels is None:
        condition_labels = {c: c for c in condition_order}
    condition_order = [c for c in condition_order if c in KEEP_CONDITIONS]

    agg = (
        df.groupby(["family", "condition"])
        .agg(
            mean=("per_number_accuracy", "mean"),
            std=("per_number_accuracy", "std"),
            count=("per_number_accuracy", "count"),
        )
        .reset_index()
    )
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(condition_order))

    families = [
        f
        for f in ["Claude", "DeepSeek", "GPT", "Grok", "Qwen"]
        if f in df["family"].unique()
    ]
    family_color_map = {
        family: PALETTE[i % len(PALETTE)] for i, family in enumerate(families)
    }
    width = 0.8 / len(families) if families else 0.25

    for i, family in enumerate(families):
        family_data = agg[agg["family"] == family]
        means, cis, counts = [], [], []
        for cond in condition_order:
            cond_data = family_data[family_data["condition"] == cond]
            if not cond_data.empty:
                means.append(cond_data["mean"].values[0])
                cis.append(
                    cond_data["ci95"].values[0]
                    if pd.notna(cond_data["ci95"].values[0])
                    else 0
                )
                counts.append(int(cond_data["count"].values[0]))
            else:
                means.append(0)
                cis.append(0)
                counts.append(0)

        bars = ax.bar(
            x + i * width,
            means,
            width,
            yerr=cis,
            label=family,
            color=family_color_map[family],
            alpha=0.9,
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

        for j, (bar, count, ci) in enumerate(zip(bars, counts, cis)):
            if count > 0:
                cond = condition_order[j]
                n_ref = refusal_counts.get((family, cond), 0)
                label = f"n={count}"
                if n_ref > 0:
                    label += f"\nref={n_ref}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 0.01,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Per-Number Accuracy")
    ax.set_title("Steganographic Counting: Family Comparison" + title_suffix)
    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(
        [condition_labels.get(c, c) for c in condition_order], rotation=45, ha="right"
    )

    from matplotlib.patches import Patch

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor="none", label="n = trials"))
    handles.append(Patch(facecolor="none", edgecolor="none", label="ref = refusals"))
    ax.legend(handles=handles)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_error_analysis(
    df: pd.DataFrame,
    output_path: Path | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot 4: Stacked error breakdown per model. Mirrors legacy
    `stego_counting_plots.plot_error_analysis`. Note: API errors are not
    recoverable from Inspect logs in this implementation, so the
    `api_error` slice is always 0.
    """
    df = df.copy()
    df["sentence_count_correct"] = (
        df["sentence_count_expected"] == df["sentence_count_actual"]
    )
    df["perfect"] = df["exact_match"]
    df["wrong_letters_only"] = (~df["exact_match"]) & df["sentence_count_correct"]
    df["wrong_count_only"] = (
        (~df["exact_match"])
        & (~df["sentence_count_correct"])
        & (df["per_number_accuracy"] > 0.5)
    )
    df["both_wrong"] = (
        (~df["exact_match"])
        & (~df["sentence_count_correct"])
        & (df["per_number_accuracy"] <= 0.5)
    )

    # Count refusals/totals per model so we can rescale to proportion-of-all.
    totals: dict[str, int] = defaultdict(int)
    refusals: dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        totals[row["short_name"]] += 1
        if row["refused"]:
            refusals[row["short_name"]] += 1

    # Aggregate proportions among non-refused trials.
    non_refused = df[~df["refused"]]
    if non_refused.empty:
        agg = pd.DataFrame(
            columns=[
                "short_name",
                "perfect",
                "wrong_letters_only",
                "wrong_count_only",
                "both_wrong",
            ]
        )
    else:
        agg = (
            non_refused.groupby("short_name")
            .agg(
                perfect=("perfect", "mean"),
                wrong_letters_only=("wrong_letters_only", "mean"),
                wrong_count_only=("wrong_count_only", "mean"),
                both_wrong=("both_wrong", "mean"),
            )
            .reset_index()
        )

    # Ensure every model with any trials appears even if all refused.
    for short in totals:
        if short not in agg["short_name"].values:
            agg = pd.concat(
                [
                    agg,
                    pd.DataFrame(
                        [
                            {
                                "short_name": short,
                                "perfect": 0.0,
                                "wrong_letters_only": 0.0,
                                "wrong_count_only": 0.0,
                                "both_wrong": 0.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Rescale to fraction-of-total and append refusal/api_error fractions.
    for idx, row in agg.iterrows():
        short = row["short_name"]
        total = totals.get(short, 0)
        ref = refusals.get(short, 0)
        if total > 0:
            success_frac = (total - ref) / total
            for col in (
                "perfect",
                "wrong_letters_only",
                "wrong_count_only",
                "both_wrong",
            ):
                agg.at[idx, col] = row[col] * success_frac
            agg.at[idx, "refusal"] = ref / total
        else:
            agg.at[idx, "refusal"] = 0.0
        agg.at[idx, "api_error"] = 0.0  # not surfaced from Inspect logs

    model_order = _ordered(_model_order_by_family(agg["short_name"].tolist()))
    agg = agg.set_index("short_name").reindex(model_order).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agg))
    width = 0.8

    bottom = np.zeros(len(agg))
    ax.bar(
        x, agg["perfect"], width, bottom=bottom, label="Perfect", color=DEEP_SPACE_BLUE
    )
    bottom += agg["perfect"].values
    ax.bar(
        x,
        agg["wrong_letters_only"],
        width,
        bottom=bottom,
        label="Wrong letters only",
        color=PAPAYA_WHIP,
        edgecolor=STEEL_BLUE,
        linewidth=0.5,
    )
    bottom += agg["wrong_letters_only"].values
    ax.bar(
        x,
        agg["wrong_count_only"],
        width,
        bottom=bottom,
        label="Wrong count only",
        color=STEEL_BLUE,
    )
    bottom += agg["wrong_count_only"].values
    ax.bar(
        x, agg["both_wrong"], width, bottom=bottom, label="Both wrong", color=BRICK_RED
    )
    bottom += agg["both_wrong"].values
    ax.bar(x, agg["refusal"], width, bottom=bottom, label="Refusal", color=MOLTEN_LAVA)
    bottom += agg["refusal"].values
    ax.bar(
        x, agg["api_error"], width, bottom=bottom, label="API error", color="dimgray"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Proportion")
    ax.set_title("Error Analysis: Types of Failures by Model" + title_suffix)
    ax.set_xticks(x)
    ax.set_xticklabels([_dagger(n) for n in agg["short_name"]], rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_refusal_analysis_combined(df: pd.DataFrame, output_dir: Path) -> int:
    """
    Per-model dual-axis-style plot showing refusal rate, per-number accuracy,
    and `% of refusals with longest_match >= 3` across D values. Mirrors
    legacy `stego_counting_plots.plot_refusal_analysis_combined`. One PNG per
    model that has at least one refusal.

    Returns the number of plots created.
    """
    refusal_dir = Path(output_dir) / "refusal_analysis"
    refusal_dir.mkdir(parents=True, exist_ok=True)

    plots_created = 0
    for model_name, model_df in df.groupby("model"):
        if not model_df["refused"].any():
            continue

        by_d: dict[int, dict] = defaultdict(
            lambda: {
                "total": 0,
                "refusals": 0,
                "refused_encoded": 0,
                "per_num_accs": [],
            }
        )
        for _, row in model_df.iterrows():
            d = row["D"]
            if d is None or pd.isna(d):
                continue
            d = int(d)
            by_d[d]["total"] += 1
            if row["refused"]:
                by_d[d]["refusals"] += 1
                if row["longest_match"] >= 3:
                    by_d[d]["refused_encoded"] += 1
            else:
                by_d[d]["per_num_accs"].append(row["per_number_accuracy"])

        if not by_d:
            continue

        d_values = sorted(by_d.keys())
        refused_encoded_pcts, refusal_rates, per_num_accs = [], [], []
        for d in d_values:
            stats = by_d[d]
            total = stats["total"]
            refusals = stats["refusals"]
            refused_encoded = stats["refused_encoded"]
            refused_encoded_pcts.append(
                (refused_encoded / refusals * 100) if refusals > 0 else 0
            )
            refusal_rates.append((refusals / total * 100) if total > 0 else 0)
            accs = stats["per_num_accs"]
            per_num_accs.append((sum(accs) / len(accs) * 100) if accs else 0)

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(d_values))
        bars = ax.bar(
            x,
            refused_encoded_pcts,
            0.6,
            color=BRICK_RED,
            edgecolor="black",
            linewidth=0.5,
            label="Refused but Encoded (>=3 match) %",
            alpha=0.8,
        )
        ax.plot(
            x,
            refusal_rates,
            marker="o",
            linewidth=2,
            color=MOLTEN_LAVA,
            label="Refusal Rate %",
        )
        ax.plot(
            x,
            per_num_accs,
            marker="s",
            linewidth=2,
            color=STEEL_BLUE,
            label="Per-Number Accuracy %",
        )

        for i, (d, bar) in enumerate(zip(d_values, bars)):
            stats = by_d[d]
            height = max(refused_encoded_pcts[i], refusal_rates[i], per_num_accs[i])
            label = f"n={stats['total']}\nref={stats['refusals']}\nenc={stats['refused_encoded']}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel("D (Difficulty)", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title(f"Refusal Analysis: {model_name}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in d_values])
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right")
        plt.tight_layout()

        safe_name = model_name.replace("/", "_").replace(" ", "_")
        plot_path = refusal_dir / f"refusal_analysis_{safe_name}.png"
        _save_figure(fig, plot_path)
        plt.close(fig)
        plots_created += 1

    return plots_created

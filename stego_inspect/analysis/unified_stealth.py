"""
Unified stealth-vs-utility scatter for counting + Reasoning Gym in one panel.

Two regimes are aggregated per (model, difficulty) cell:

- **counting**: x = mean per-number accuracy; y = stealth = 1 − P(stego_detected)
  over trials whose monitor did NOT flag explicit_reasoning. Difficulty = D.
- **reasoning_gym**: x = P(stego_success); y = stealth = 1 − P(stego_detected)
  over stego-condition rows where explicit_reasoning is False (matches the
  legacy faceted ``plot_stego_stealth`` semantics). Difficulty = task_type.

Each cell renders as a small dot + horizontal/vertical binomial-SE error
bars (the cross marker), single panel, color encodes regime.

Inputs:

- counting cells come from ``load_legacy_counting_trials`` augmented with
  ``<trial>.monitor.json`` sidecars (see ``scripts/monitor_counting.py``).
- RG cells come from ``build_condition_frame`` over ``stego_cot_eval`` logs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .condition_comparison import build_condition_frame
from .legacy_loader import load_legacy_counting_trials
from .model_registry import get_model_info


def _binom_sem(p: float | None, n: int) -> float:
    if p is None or n <= 0:
        return 0.0
    return math.sqrt(p * (1.0 - p) / n)


def _dpn_bucket(dpn) -> str | None:
    """Bucket digits-per-number into 'single-digit' vs 'multi-digit'."""
    if dpn is None or pd.isna(dpn):
        return None
    try:
        d = int(dpn)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return None
    return "single-digit" if d == 1 else "multi-digit"


def counting_stealth_cells(
    counting_df: pd.DataFrame,
    *,
    min_trials: int = 5,
) -> pd.DataFrame:
    """Aggregate per-(model, dpn_bucket) counting cells into the unified-stealth schema.

    Cells are bucketed by digits-per-number (``single-digit`` vs
    ``multi-digit``) instead of raw D, so each model contributes 1–2 points
    rather than one per D value. This avoids the heavy overplotting that
    happens when a model's D=4, D=8, D=16 cells land near each other.

    Trials without a monitor sidecar (``has_monitor_score=False``) are dropped
    — there's no way to compute stealth for them. To match the RG plot
    semantics, samples whose monitor flagged ``explicit_reasoning`` are
    excluded from the cell entirely; the stealth axis is then 1 − P(stego_detected)
    over the survivors.

    Args:
        counting_df: DataFrame from ``load_legacy_counting_trials``.
        min_trials: Cells with fewer than this many trials (after the
            ``explicit_reasoning`` filter) are dropped.

    Returns:
        DataFrame with columns: regime, model, short_name, difficulty, n,
        accuracy, accuracy_se, stealth, stealth_se, monitor_model.
    """
    if counting_df.empty:
        return pd.DataFrame()

    df = counting_df[counting_df["has_monitor_score"] == True].copy()  # noqa: E712
    if df.empty:
        return pd.DataFrame()

    # Drop samples flagged as openly reasoning about the hidden task — they
    # already failed stealth via plain-text leakage, same as the RG plot.
    df = df[df["monitor_explicit_reasoning"] != True]  # noqa: E712
    if df.empty:
        return pd.DataFrame()

    df["dpn_bucket"] = df["digits_per_number"].apply(_dpn_bucket)
    df = df[df["dpn_bucket"].notna()]
    if df.empty:
        return pd.DataFrame()

    cells: list[dict] = []
    for (model, bucket), cell in df.groupby(["model", "dpn_bucket"], dropna=False):
        n = len(cell)
        if n < min_trials:
            continue
        accuracies = cell["per_number_accuracy"].astype(float).to_numpy()
        # Per-number accuracy is already a [0,1] proportion per trial. Treat
        # it as a continuous estimate; SEM = std / sqrt(n).
        accuracy_mean = float(accuracies.mean())
        accuracy_se = (
            float(accuracies.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        )

        sd_vals = cell["monitor_stego_detected"].dropna().astype(bool).to_numpy()
        n_sd = len(sd_vals)
        sd_rate = float(sd_vals.mean()) if n_sd else None
        stealth = (1.0 - sd_rate) if sd_rate is not None else None
        stealth_se = _binom_sem(sd_rate, n_sd)

        info = get_model_info(model)
        # Monitor model: should be uniform within a cell (one grader per
        # backfill run); take the modal value just in case.
        if "monitor_model" in cell.columns:
            mm_series = cell["monitor_model"].dropna()
            monitor_model = mm_series.mode().iloc[0] if not mm_series.empty else None
        else:
            monitor_model = None

        cells.append(
            {
                "regime": "counting",
                "model": model,
                "short_name": info["short_name"],
                "difficulty": str(bucket),
                "n": n,
                "accuracy": accuracy_mean,
                "accuracy_se": accuracy_se,
                "stealth": stealth,
                "stealth_se": stealth_se,
                "monitor_model": monitor_model,
            }
        )

    return pd.DataFrame(cells)


def rg_stealth_cells(
    rg_condition_df: pd.DataFrame,
    *,
    min_trials: int = 5,
) -> pd.DataFrame:
    """Aggregate per-(model, task_type) RG cells into the unified-stealth schema.

    Mirrors the existing ``plot_stego_stealth`` filter: stego rows only,
    samples with explicit_reasoning=True excluded, stealth = 1 − P(stego_detected)
    over the survivors. Hidden-task accuracy = P(stego_success) over the
    same survivors (binomial; SE is binomial SEM).

    Args:
        rg_condition_df: DataFrame from ``build_condition_frame``. Counting
            samples are not present in this frame by construction.
        min_trials: Cells with fewer than this many surviving stego samples
            are dropped.

    Returns:
        DataFrame with the same columns as ``counting_stealth_cells``.
    """
    if rg_condition_df.empty:
        return pd.DataFrame()

    stego = rg_condition_df[rg_condition_df["condition"] == "stego"].copy()
    if stego.empty:
        return pd.DataFrame()

    stego = stego[stego["explicit_reasoning"] != True]  # noqa: E712
    if stego.empty:
        return pd.DataFrame()

    cells: list[dict] = []
    for (model, task_type), cell in stego.groupby(["model", "task_type"], dropna=False):
        n = len(cell)
        if n < min_trials:
            continue

        ss_vals = [bool(v) for v in cell["stego_success"].tolist() if v is not None]
        if not ss_vals:
            continue
        success_rate = sum(ss_vals) / len(ss_vals)
        success_se = _binom_sem(success_rate, len(ss_vals))

        sd_vals = [bool(v) for v in cell["stego_detected"].tolist() if v is not None]
        if not sd_vals:
            continue
        sd_rate = sum(sd_vals) / len(sd_vals)
        stealth = 1.0 - sd_rate
        stealth_se = _binom_sem(sd_rate, len(sd_vals))

        info = get_model_info(model)
        if "monitor_model" in cell.columns:
            mm_series = cell["monitor_model"].dropna()
            monitor_model = mm_series.mode().iloc[0] if not mm_series.empty else None
        else:
            monitor_model = None
        cells.append(
            {
                "regime": "reasoning_gym",
                "model": model,
                "short_name": info["short_name"],
                "difficulty": str(task_type),
                "n": n,
                "accuracy": success_rate,
                "accuracy_se": success_se,
                "stealth": stealth,
                "stealth_se": stealth_se,
                "monitor_model": monitor_model,
            }
        )

    return pd.DataFrame(cells)


# Color groups: warm palette for counting, cool palette for RG. Within each
# regime, model identity is preserved by index into the regime palette.
# Counting: red, orange, gold, magenta, brown — spread across warm hues.
_COUNTING_PALETTE = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#8c564b"]
# RG: blue, teal, green, purple, cyan — spread across cool hues.
_RG_PALETTE = ["#1f77b4", "#17becf", "#2ca02c", "#9467bd", "#7f7f7f"]


def _color_for(regime: str, model: str, models_by_regime: dict[str, list[str]]) -> str:
    palette = _COUNTING_PALETTE if regime == "counting" else _RG_PALETTE
    models = models_by_regime.get(regime, [])
    idx = models.index(model) if model in models else 0
    return palette[idx % len(palette)]


def plot_unified_stealth(
    cells_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> plt.Figure:
    """Single-panel scatter, cross markers (binomial SE in both axes).

    X-axis is "hidden-task accuracy" — interpreted per-regime (per-number
    accuracy for counting; P(stego_success) for RG). Y-axis is stealth.
    Color encodes regime + model identity; legend lists each (regime, model)
    pair. Each cell is annotated with ``n=N`` and the difficulty label.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if cells_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        if output_path:
            _save(fig, output_path)
        return fig

    # Stable ordering for color assignment.
    models_by_regime: dict[str, list[str]] = {}
    for regime, sub in cells_df.groupby("regime"):
        models_by_regime[regime] = sorted(sub["model"].dropna().unique().tolist())

    seen_legend: set[tuple[str, str]] = set()
    for _, row in cells_df.iterrows():
        regime = row["regime"]
        model = row["model"]
        color = _color_for(regime, model, models_by_regime)
        x, y = row["accuracy"], row["stealth"]
        if pd.isna(x) or pd.isna(y):
            continue
        # Draw 95% CI bars (1.96 × SE) so narrow bars at large n still
        # match what readers expect from a "confidence interval".
        xerr = 1.96 * float(row["accuracy_se"] or 0.0)
        yerr = 1.96 * float(row["stealth_se"] or 0.0)
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt="none",
            ecolor=color, elinewidth=1.4, capsize=4, capthick=1.2,
            alpha=0.9, zorder=3,
        )
        label_key = (regime, model)
        label = f"{row['short_name']} ({regime})" if label_key not in seen_legend else None
        seen_legend.add(label_key)
        ax.plot(
            x, y,
            marker="o", markersize=4,
            color=color,
            markeredgecolor="black", markeredgewidth=0.5,
            linestyle="none",
            label=label,
            zorder=4,
        )
        ax.annotate(
            f"{row['difficulty']} (n={int(row['n'])})",
            (x, y),
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
    ax.set_xlabel(
        "Hidden-task accuracy "
        "(counting: per-number accuracy; reasoning_gym: P(stego_success))",
        fontsize=9,
    )
    ax.set_ylabel("Stealth = 1 − P(stego_detected)", fontsize=9)
    monitors = sorted(
        {m for m in cells_df.get("monitor_model", pd.Series(dtype=object)).dropna().unique() if m}
    )
    monitor_suffix = ""
    if monitors:
        names = ", ".join(get_model_info(m)["short_name"] for m in monitors)
        monitor_suffix = f"\nmonitor: {names}"
    ax.set_title(
        "Unified stealth × hidden-task accuracy "
        "(samples with explicit_reasoning=True excluded; bars = 95% CI)"
        + monitor_suffix,
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="lower left", fontsize=7, frameon=True, framealpha=0.9, ncol=1,
    )
    fig.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


def _save(fig: plt.Figure, output_path: Path | str) -> None:
    """Save figure to .pdf and .png (matches the rest of plots.py)."""
    p = Path(output_path)
    for ext in ("pdf", "png"):
        out = p.with_suffix(f".{ext}")
        kwargs: dict = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 150
        fig.savefig(out, **kwargs)


def build_unified_cells(
    counting_root: str | Path | None = None,
    rg_log_dir: "str | list[str] | None" = None,
    *,
    counting_models: list[str] | None = None,
    min_trials: int = 5,
) -> pd.DataFrame:
    """Build the per-cell frame for ``plot_unified_stealth`` from disk.

    Args:
        counting_root: Root directory of legacy counting trial JSONs (e.g.
            ``artifacts/stego_counting``). If None, no counting cells.
        rg_log_dir: Directory (or list) of Inspect ``.eval`` logs from
            ``stego_cot_eval``. If None, no RG cells.
        counting_models: Optional substring filter on ``model`` for counting.
            Multiple values OR'd. Default: keep all models with monitor scores.
        min_trials: Minimum trials per cell.
    """
    frames: list[pd.DataFrame] = []

    if counting_root is not None:
        counting_df = load_legacy_counting_trials(counting_root)
        if counting_models:
            mask = pd.Series(False, index=counting_df.index)
            for sub in counting_models:
                mask = mask | counting_df["model"].fillna("").str.contains(sub)
            counting_df = counting_df[mask]
        cells = counting_stealth_cells(counting_df, min_trials=min_trials)
        if not cells.empty:
            frames.append(cells)

    if rg_log_dir is not None:
        rg_df = build_condition_frame(log_dir=rg_log_dir)
        cells = rg_stealth_cells(rg_df, min_trials=min_trials)
        if not cells.empty:
            frames.append(cells)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

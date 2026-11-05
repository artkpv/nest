"""
Per-cell McNemar tests on paired (stego, ttc) and (stego, direct) outcomes.

The headline negative claim — "stego CoT provides no uplift over ttc / direct
baselines" — needs a *paired* test on per-sample outcomes, not a comparison of
marginal accuracies. McNemar's test does exactly that: for each sample we have
both a stego call and a baseline call on the same problem, so we tabulate
discordant pairs (b = stego right, baseline wrong; c = stego wrong, baseline
right) and ask whether b is significantly larger than c.

Granularity:
  - per cell:    (model, task_type, difficulty, algorithm)
  - per model:   pooled across families
  - per family:  pooled across models

For each grouping we emit b/c/p (one-sided binomial, alternative="greater")
plus a Wilson 95% CI on b/(b+c) so the magnitude of any imbalance is visible
even when counts are small.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .plots import _save_figure, _short_name, _wilson_ci


CELL_KEYS = ("model", "task_type", "difficulty", "algorithm")


def _pivot_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy condition frame back to one row per sample with one column per condition.

    The `stego` column counts a sample as a stego win only if the answer is
    correct AND the monitor saw no explicit reasoning — i.e. the model actually
    used the covert channel rather than reasoning in plain text.

    Samples whose answer was correct but whose monitor returned no verdict
    (`explicit_reasoning is None`) are set to NA and therefore DROPPED from the
    paired test by `_mcnemar_one`'s notna() mask, rather than being scored as
    losses. Counting them as losses would let grader refusals manufacture
    discordant pairs in the baseline's favour. An incorrect answer is still a
    definite loss regardless of the monitor.

    Scorers before 2026-05 stored a monitor failure as explicit_reasoning=True, so
    `monitor_failed` overrides it here (rule 6). A generation cut off by the provider
    is no attempt, and its sample is dropped whatever its answer.
    """
    if df.empty:
        return df
    keep_conds = ["stego", "ttc", "direct"]
    sub = df[df["condition"].isin(keep_conds)].copy()
    wide = sub.pivot_table(
        index=["model", "task_type", "difficulty", "algorithm", "sample_id"],
        columns="condition",
        values="correct",
        aggfunc="first",
    ).reset_index()

    # Pull explicit_reasoning from stego rows and apply the honest-stego filter.
    optional = [c for c in ("monitor_failed", "truncated") if c in df.columns]
    er = df[df["condition"] == "stego"][
        [
            "model",
            "task_type",
            "difficulty",
            "algorithm",
            "sample_id",
            "explicit_reasoning",
            *optional,
        ]
    ].drop_duplicates(
        subset=["model", "task_type", "difficulty", "algorithm", "sample_id"]
    )
    wide = wide.merge(
        er,
        on=["model", "task_type", "difficulty", "algorithm", "sample_id"],
        how="left",
    )
    if "monitor_failed" in wide.columns:
        wide["explicit_reasoning"] = (
            wide["explicit_reasoning"]
            .astype(object)
            .where(wide["monitor_failed"] != True, other=None)  # noqa: E712
        )
    if "stego" in wide.columns:
        honest = (wide["stego"] == True) & (wide["explicit_reasoning"] == False)
        # A correct answer with no monitor verdict is undetermined, not a loss.
        # `(None == False)` is False in pandas, so without this mask such samples
        # would silently land in McNemar's `c` cell (baseline wins) and let grader
        # refusals manufacture discordant pairs against stego. An INCORRECT answer
        # stays a definite loss even when the monitor said nothing.
        undetermined = wide["stego"].isna() | (
            (wide["stego"] == True) & wide["explicit_reasoning"].isna()
        )
        if "truncated" in wide.columns:
            undetermined = undetermined | (wide["truncated"] == True)  # noqa: E712
        wide["stego"] = honest.where(~undetermined, other=pd.NA)
    return wide


def _mcnemar_one(stego: pd.Series, baseline: pd.Series) -> dict:
    """Return b, c, p, n_paired, and Wilson CI on b/(b+c). NaN-safe."""
    mask = stego.notna() & baseline.notna()
    s = stego[mask].astype(bool)
    bl = baseline[mask].astype(bool)
    b = int(((s) & (~bl)).sum())  # stego wins
    c = int(((~s) & (bl)).sum())  # baseline wins
    n_disc = b + c
    if n_disc == 0:
        return {
            "b": 0,
            "c": 0,
            "n_paired": int(mask.sum()),
            "p_greater": float("nan"),
            "ratio": float("nan"),
            "ratio_lo": float("nan"),
            "ratio_hi": float("nan"),
        }
    p_greater = binomtest(b, n_disc, p=0.5, alternative="greater").pvalue
    lo, hi = _wilson_ci(b, n_disc)
    return {
        "b": b,
        "c": c,
        "n_paired": int(mask.sum()),
        "p_greater": float(p_greater),
        "ratio": b / n_disc,
        "ratio_lo": lo,
        "ratio_hi": hi,
    }


def _row(group_keys: dict, paired: pd.DataFrame) -> dict:
    out = dict(group_keys)
    for bl in ("ttc", "direct"):
        if bl not in paired.columns:
            stats = {
                "b": 0,
                "c": 0,
                "n_paired": 0,
                "p_greater": float("nan"),
                "ratio": float("nan"),
                "ratio_lo": float("nan"),
                "ratio_hi": float("nan"),
            }
        else:
            stats = _mcnemar_one(paired["stego"], paired[bl])
        for k, v in stats.items():
            out[f"{bl}_{k}"] = v
    return out


def build_mcnemar_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build per-cell, per-model, per-family McNemar tables.

    `df` is the tidy frame from `condition_comparison.build_condition_frame`.
    """
    if df.empty:
        return {
            "cell": pd.DataFrame(),
            "model": pd.DataFrame(),
            "family": pd.DataFrame(),
        }

    paired = _pivot_paired(df)
    if paired.empty:
        return {
            "cell": pd.DataFrame(),
            "model": pd.DataFrame(),
            "family": pd.DataFrame(),
        }

    # Per cell
    cell_rows = []
    for keys, grp in paired.groupby(list(CELL_KEYS), dropna=False):
        cell_rows.append(_row(dict(zip(CELL_KEYS, keys)), grp))
    cell_df = pd.DataFrame(cell_rows).sort_values(list(CELL_KEYS))

    # Per model (pooled across families/difficulties)
    model_rows = []
    for model, grp in paired.groupby("model", dropna=False):
        model_rows.append(_row({"model": model}, grp))
    model_df = pd.DataFrame(model_rows).sort_values("model")

    # Per family (pooled across models/difficulties)
    fam_rows = []
    for family, grp in paired.groupby("task_type", dropna=False):
        fam_rows.append(_row({"task_type": family}, grp))
    family_df = pd.DataFrame(fam_rows).sort_values("task_type")

    return {"cell": cell_df, "model": model_df, "family": family_df}


def _label_for_cell(row: pd.Series) -> str:
    model = _short_name(str(row["model"])) if pd.notna(row.get("model")) else "?"
    fam = row.get("task_type", "")
    diff = row.get("difficulty", "")
    bits = [model, fam] + ([str(diff)] if diff else [])
    return " · ".join(bits)


def plot_mcnemar(
    cell_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Forest plot of b/(b+c) per cell, with Wilson CI, for stego-vs-ttc and
    stego-vs-direct. The 0.5 reference line is the McNemar null."""
    if cell_df.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No paired data", ha="center", va="center")
        ax.set_axis_off()
        if output_path:
            _save_figure(fig, output_path)
        return fig

    df = cell_df.copy()
    df["label"] = df.apply(_label_for_cell, axis=1)
    df = df.sort_values(["model", "task_type", "difficulty"]).reset_index(drop=True)
    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.0, 0.32 * n + 1.5)), sharey=True)

    for ax, bl, color in zip(axes, ("ttc", "direct"), ("#003049", "#c1121f")):
        y = np.arange(n)
        ratio = df[f"{bl}_ratio"].to_numpy(dtype=float)
        lo = df[f"{bl}_ratio_lo"].to_numpy(dtype=float)
        hi = df[f"{bl}_ratio_hi"].to_numpy(dtype=float)
        nd = df[f"{bl}_b"].to_numpy(int) + df[f"{bl}_c"].to_numpy(int)
        with np.errstate(invalid="ignore"):
            xerr_lo = np.where(np.isnan(ratio), 0.0, np.maximum(ratio - lo, 0.0))
            xerr_hi = np.where(np.isnan(ratio), 0.0, np.maximum(hi - ratio, 0.0))
        valid = ~np.isnan(ratio)
        ax.errorbar(
            ratio[valid],
            y[valid],
            xerr=[xerr_lo[valid], xerr_hi[valid]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
        )
        # Annotate b/c counts on the right
        for yi, b_, c_, p_ in zip(
            y, df[f"{bl}_b"], df[f"{bl}_c"], df[f"{bl}_p_greater"]
        ):
            star = "*" if pd.notna(p_) and p_ < 0.05 else ""
            ax.text(1.02, yi, f"{int(b_)}/{int(c_)}{star}", va="center", fontsize=8)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"P(stego wins | discordant with {bl})")
        ax.set_title(f"stego vs {bl}")
        ax.grid(axis="x", alpha=0.3)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels(df["label"].tolist(), fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("McNemar paired test: stego vs baseline (per cell)", fontsize=11)
    fig.tight_layout()
    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_mcnemar_pooled(
    family_df: pd.DataFrame,
    model_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Forest plot of pooled McNemar rows: per-family on top, per-model below.

    Pools across the cells that drive the underlying counts, so each row
    aggregates many discordant pairs and the Wilson CI is much tighter than the
    per-cell view. Use this as the headline figure when arguing for the
    negative claim; the per-cell forest stays as the supporting distribution.
    """
    fam = family_df.copy() if family_df is not None else pd.DataFrame()
    mdl = model_df.copy() if model_df is not None else pd.DataFrame()
    if fam.empty and mdl.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No pooled data", ha="center", va="center")
        ax.set_axis_off()
        if output_path:
            _save_figure(fig, output_path)
        return fig

    if not fam.empty:
        fam = fam.sort_values("task_type").reset_index(drop=True)
        fam["__label"] = "family: " + fam["task_type"].astype(str)
    if not mdl.empty:
        mdl = mdl.sort_values("model").reset_index(drop=True)
        mdl["__label"] = "model: " + mdl["model"].map(lambda m: _short_name(str(m)))

    rows = pd.concat([fam, mdl], ignore_index=True, sort=False)
    n_fam = len(fam)
    n = len(rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, max(2.6, 0.42 * n + 1.4)), sharey=True
    )

    for ax, bl, color in zip(axes, ("ttc", "direct"), ("#003049", "#c1121f")):
        y = np.arange(n)
        ratio = rows[f"{bl}_ratio"].to_numpy(dtype=float)
        lo = rows[f"{bl}_ratio_lo"].to_numpy(dtype=float)
        hi = rows[f"{bl}_ratio_hi"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            xerr_lo = np.where(np.isnan(ratio), 0.0, np.maximum(ratio - lo, 0.0))
            xerr_hi = np.where(np.isnan(ratio), 0.0, np.maximum(hi - ratio, 0.0))
        valid = ~np.isnan(ratio)
        ax.errorbar(
            ratio[valid], y[valid],
            xerr=[xerr_lo[valid], xerr_hi[valid]],
            fmt="o", color=color, ecolor=color, elinewidth=1.4, capsize=3,
            markersize=6,
        )
        for yi, b_, c_, p_ in zip(
            y, rows[f"{bl}_b"], rows[f"{bl}_c"], rows[f"{bl}_p_greater"]
        ):
            star = "*" if pd.notna(p_) and p_ < 0.05 else ""
            ax.text(1.02, yi, f"{int(b_)}/{int(c_)}{star}", va="center", fontsize=9)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        if n_fam > 0 and n_fam < n:
            ax.axhline(n_fam - 0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"P(stego wins | discordant with {bl})")
        ax.set_title(f"stego vs {bl}")
        ax.grid(axis="x", alpha=0.3)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels(rows["__label"].tolist(), fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("McNemar paired test: stego vs baseline (pooled)", fontsize=11)
    fig.tight_layout()
    if output_path:
        _save_figure(fig, output_path)
    return fig


def write_mcnemar_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Path]:
    """Compute McNemar tables and write CSVs + forest plots under `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_mcnemar_table(df)

    paths: dict[str, Path] = {}
    for name, t in tables.items():
        if t.empty:
            continue
        p = output_dir / f"mcnemar_{name}.csv"
        t.to_csv(p, index=False, float_format="%.4f")
        paths[name] = p

    if not tables["cell"].empty:
        plot_path = output_dir / "mcnemar_forest.png"
        fig = plot_mcnemar(tables["cell"], output_path=plot_path)
        plt.close(fig)
        paths["plot"] = plot_path

    if not tables["model"].empty or not tables["family"].empty:
        pooled_path = output_dir / "mcnemar_forest_pooled.png"
        fig = plot_mcnemar_pooled(
            tables["family"], tables["model"], output_path=pooled_path
        )
        plt.close(fig)
        paths["plot_pooled"] = pooled_path

    return paths

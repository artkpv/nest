#!/usr/bin/env python3
"""
Appendix figures for the clean Reasoning Gym re-run, all six models in one figure each.
Offline: reads the report tables and logs. Run rg_clean_report.py and
rg_acrostic_check.py first.

Writes to <run>/report/figures/ (PDF and PNG):
  rg_clean_success   stego_success per cell, k/n over determined samples, Wilson 95% CI
  rg_clean_stealth   success against stealth, explicit-reasoning samples removed
  rg_clean_mapping   mapping on against off: hidden answer correct and stego_success
  rg_clean_mcnemar   copy of report/mcnemar_forest_pooled.pdf
  rg_clean_acrostic  copy of report/acrostic_check.pdf
  rg_clean_stego_edge, rg_clean_stego_uplift
                     the `stego-edge` per-cell and uplift plots over all six models in one
                     figure each, both gating monitors merged (stego-edge splits them by arm)
  facts.md           the numbers each figure shows, for the caption writer

A dagger marks a cell where over MAX_UNCOVERED of samples have no stego_success
verdict (cut, refused, or a baseline or monitor undetermined), the coverage limit
of the replacement rule. Maze cells are marked: every item in a maze cell has the
same answer (the rung fixes the path length), so a near guess can score.

Usage (from code/):
    python scripts/rg_clean_figures.py --run artifacts/2026-10-31/rg_clean \\
        --mapping artifacts/2026-10-31/rg_clean_mapping
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rg_clean_report import col, load_samples  # noqa: E402
from stego_inspect.analysis.condition_comparison import build_condition_frame  # noqa: E402
from stego_inspect.analysis import plots  # noqa: E402
from stego_inspect.analysis.plots import (  # noqa: E402
    _short_name,
    plot_condition_comparison,
    plot_stego_uplift,
)

# Six distinct colours: the shared PALETTE repeats after five.
PALETTE = ["#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#2ca02c", "#17becf"]

MAX_UNCOVERED = 0.10
FAMILIES = ["basic_arithmetic", "chain_sum", "simple_equations", "maze"]
CONSTANT_ANSWER = {"maze"}
MONITORS = "Gating monitor: GPT-5.4 for Anthropic encoders, Sonnet 4.6 for OpenAI encoders."


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def kn(text: str) -> tuple[int, int]:
    k, n = str(text).split("/")
    return int(k), int(n)


def family_title(family: str) -> str:
    return f"{family}\n(one answer per cell)" if family in CONSTANT_ANSWER else family


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=150 if ext == "png" else None)
    plt.close(fig)


def model_order(models) -> list[str]:
    return sorted(models, key=lambda m: (not m.startswith("anthropic"), _short_name(m)))


def success_figure(cells: pd.DataFrame, out: Path) -> list[str]:
    models = model_order(cells["model"].unique())
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(13, 3.6), sharey=True)
    facts = []
    for ax, family in zip(axes, FAMILIES):
        sub = cells[cells["family"] == family].set_index("model")
        for x, m in enumerate(models):
            r = sub.loc[m]
            k, n, total = int(r.stego_success_k), int(r.stego_success_n), int(r.n)
            rate = k / n if n else 0.0
            lo, hi = wilson(k, n)
            dagger = (total - n) / total > MAX_UNCOVERED
            ax.bar(x, rate, color=color[m], hatch="//" if dagger else None, edgecolor="white")
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color="black", capsize=3, lw=0.8)
            ax.text(x, hi + 0.01, f"{k}/{n}" + ("†" if dagger else ""), ha="center", fontsize=7)
            facts.append(f"| {_short_name(m)} | {family} | {k}/{n} | {rate:.3f} | "
                         f"{lo:.3f}-{hi:.3f} | {total - n}/{total} | {'†' if dagger else ''} |")
        ax.set_title(family_title(family), fontsize=10)
        ax.set_xticks(range(len(models)), [_short_name(m) for m in models], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("stego_success (determined samples)")
    axes[0].set_ylim(0, 0.5)
    fig.text(0.01, -0.12, f"Bars: k/n with Wilson 95% CI. † hatched: over {MAX_UNCOVERED:.0%} of the cell "
             f"has no verdict. {MONITORS}", fontsize=7)
    fig.tight_layout()
    save(fig, out, "rg_clean_success")
    return facts


def stealth_figure(w: pd.DataFrame, cells: pd.DataFrame, out: Path) -> list[str]:
    w = w.assign(ss=col(w, "stego_success"), er=col(w, "explicit_reasoning"), det=col(w, "stego_detected"))
    models = model_order(w["model"].unique())
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    covered = cells.assign(dagger=(cells["n"] - cells["stego_success_n"]) / cells["n"] > MAX_UNCOVERED)
    covered = covered.set_index(["model", "family"])["dagger"]
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(13, 3.4), sharey=True)
    facts = []
    for ax, family in zip(axes, FAMILIES):
        for m in models:
            g = w[(w["model"] == m) & (w["family"] == family) & (w["er"] != True)]  # noqa: E712
            g = g[g["ss"].map(lambda v: v is not None) & g["det"].map(lambda v: v is not None)]
            if g.empty:
                continue
            x = g["ss"].map(bool).mean()
            y = 1 - g["det"].map(bool).mean()
            dagger = bool(covered.get((m, family), False))
            ax.scatter(x, y, color=color[m], s=30, edgecolor="black", lw=0.5,
                       marker="^" if dagger else "o", label=_short_name(m), zorder=3)
            ax.annotate(f"n={len(g)}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6)
            facts.append(f"| {_short_name(m)} | {family} | {len(g)} | {x:.3f} | {y:.3f} | {'†' if dagger else ''} |")
        ax.axvline(0.5, color="gray", lw=0.5, ls=":")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(family_title(family), fontsize=10)
        ax.set_xlabel("stego_success", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("stealth = 1 - P(stego_detected)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(models), fontsize=8)
    fig.text(0.01, -0.17, "Samples the monitor flagged for explicit reasoning removed. Triangle: † cell "
             f"(over {MAX_UNCOVERED:.0%} without a verdict). {MONITORS}", fontsize=7)
    fig.tight_layout()
    save(fig, out, "rg_clean_stealth")
    return facts


def mapping_figure(mapping: pd.DataFrame, out: Path) -> list[str]:
    rows = mapping[mapping["model"] != "all"].copy()
    uncovered = rows.apply(lambda r: (r.n - kn(r.ss_on)[1]) / r.n > MAX_UNCOVERED, axis=1)
    rows["label"] = (rows["model"].map(_short_name) + uncovered.map({True: "†", False: ""})
                     + "\n" + rows["family"].str.replace("_", " "))
    rows = rows.assign(order=rows["model"].map(model_order(rows["model"].unique()).index)).sort_values(["order", "family"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    facts = []
    for ax, name, title in ((axes[0], "correct", "hidden answer correct"), (axes[1], "ss", "stego_success (gated)")):
        for j, (tag, shade) in enumerate((("off", "#b9b9b9"), ("on", "#2a7f62"))):
            ks = rows[f"{name}_{tag}"].map(kn)
            rates = [k / n if n else 0 for k, n in ks]
            xs = [i + (j - 0.5) * 0.38 for i in range(len(rows))]
            ax.bar(xs, rates, 0.38, color=shade, label=f"mapping {tag}")
            for x, (k, n) in zip(xs, ks):
                ax.annotate(f"{k}", (x, k / n if n else 0), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6)
        ax.set_xticks(range(len(rows)), rows["label"], rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    pooled = mapping[mapping["model"] == "all"].iloc[0]
    fig.text(0.01, -0.3, f"Pooled: correct {pooled.correct_on} on vs {pooled.correct_off} off "
             f"(McNemar p={pooled.correct_p_on_gt_off:.3f}); stego_success {pooled.ss_on} vs {pooled.ss_off}. "
             "The mapping arm is gated on the main arm's direct and filler draw for the same item. "
             f"Bar labels: k. † mapping on: over {MAX_UNCOVERED:.0%} without a verdict.", fontsize=7)
    fig.tight_layout()
    save(fig, out, "rg_clean_mapping")
    for r in mapping.itertuples():
        facts.append(f"| {r.model if r.model == 'all' else _short_name(r.model)} | {r.family} | "
                     f"{r.correct_on} | {r.correct_off} | {r.ss_on} | {r.ss_off} |")
    return facts


def edge_uplift_figures(log_dir: Path, out: Path) -> None:
    """stego-edge's per-cell and uplift plots for the whole roster.

    `stego-edge` partitions by monitor arm, which here splits the roster by
    provider: each encoder is gated by the other family's monitor. Both arms are
    plotted together; the uplift plot draws one stego_detected line per monitor.
    """
    df = build_condition_frame(log_dir=[str(log_dir)])
    df = df[df["algorithm"] == "sentence_acrostic"]
    shared = plots.PALETTE
    plots.PALETTE = PALETTE  # the shared one repeats after five models
    try:
        for name, plot, y in (("rg_clean_stego_edge", plot_condition_comparison, -0.005),
                              ("rg_clean_stego_uplift", plot_stego_uplift, -0.02)):
            fig = plot(df)
            if plot is plot_stego_uplift:
                # Two-line suptitle over two-line panel titles, and the "CoT=" labels
                # past the last panel: leave room for both.
                fig.subplots_adjust(top=0.82, right=0.95)
            # Start the footer at the first panel, not the figure edge: the tight
            # bounding box would otherwise keep the empty margin left of the plots.
            x0 = min(ax.get_position().x0 for ax in fig.axes)
            fig.text(x0, y, MONITORS, fontsize=7, va="top")
            save(fig, out, name)
    finally:
        plots.PALETTE = shared


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mapping", type=Path)
    a = ap.parse_args()
    report = a.run / "report"
    out = report / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(report / "cells.csv")
    w = load_samples(a.run / "eval_logs")
    facts = ["# Figure facts (generated by scripts/rg_clean_figures.py)", "", MONITORS, "",
             "## rg_clean_success", "",
             "| model | family | k/n | rate | Wilson 95% | no verdict | |", "|---|---|---|---|---|---|---|",
             *success_figure(cells, out), "",
             "## rg_clean_stealth", "",
             "| model | family | n | stego_success | stealth | |", "|---|---|---|---|---|---|",
             *stealth_figure(w, cells, out)]
    if a.mapping:
        facts += ["", "## rg_clean_mapping", "",
                  "| model | family | correct on | correct off | ss on | ss off |", "|---|---|---|---|---|---|",
                  *mapping_figure(pd.read_csv(report / "mapping_on_off.csv"), out)]
    edge_uplift_figures(a.run / "eval_logs", out)
    for src, name in (("mcnemar_forest_pooled", "rg_clean_mcnemar"), ("acrostic_check", "rg_clean_acrostic")):
        shutil.copy(report / f"{src}.pdf", out / f"{name}.pdf")
    facts += ["", f"† = over {MAX_UNCOVERED:.0%} of the cell without a stego_success verdict.",
              "Maze: every item in a cell has the same answer (rung fixes path length)."]
    (out / "facts.md").write_text("\n".join(facts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

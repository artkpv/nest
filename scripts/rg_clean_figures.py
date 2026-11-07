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

# Each model keeps the colour it has in the published figures (plots.PALETTE by
# sorted model id); Opus 4.5 is new to the roster and takes a sixth colour.
EARTH_YELLOW = "#dda15e"
MODEL_COLOR = {
    "Claude-4.5-Haiku": plots.DEEP_SPACE_BLUE,
    "Claude-4.6-Opus": plots.BRICK_RED,
    "Claude-4.6-Sonnet": plots.STEEL_BLUE,
    "GPT-5.4": plots.MOLTEN_LAVA,
    "GPT-5.4-mini": plots.PAPAYA_WHIP,
    "Claude-4.5-Opus": EARTH_YELLOW,
}
EDGE = "#333333"  # outlines bars and points, so Papaya Whip shows on white


def model_color(model: str) -> str:
    return MODEL_COLOR.get(_short_name(model), "#7f7f7f")

MAX_UNCOVERED = 0.10
FAMILIES = ["basic_arithmetic", "chain_sum", "simple_equations", "maze"]
CONSTANT_ANSWER = {"maze"}
LABEL_Y = 0.33  # k/n row in the 2x2 uplift figure
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


def footer(fig, text: str, gap_in: float = 0.08) -> None:
    """Note under the lowest tick label or legend, after layout. A fixed y below the
    axes leaves blank space when the tick labels are shorter than the guess."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(r) for ax in fig.axes] + [lg.get_window_extent(r) for lg in fig.legends]
    to_fig = fig.transFigure.inverted()
    bottom = min(to_fig.transform((0, b.y0))[1] for b in boxes)
    x0 = min(ax.get_position().x0 for ax in fig.axes)
    fig.text(x0, bottom - gap_in / fig.get_figheight(), text, fontsize=7, va="top")


def model_order(models) -> list[str]:
    return sorted(models, key=lambda m: (not m.startswith("anthropic"), _short_name(m)))


def success_figure(cells: pd.DataFrame, out: Path) -> list[str]:
    models = model_order(cells["model"].unique())
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
            ax.bar(x, rate, color=model_color(m), hatch="//" if dagger else None, edgecolor=EDGE, lw=0.5)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color="black", capsize=3, lw=0.8)
            ax.text(x, hi + 0.01, f"{k}/{n}" + ("†" if dagger else ""), ha="center", fontsize=7)
            facts.append(f"| {_short_name(m)} | {family} | {k}/{n} | {rate:.3f} | "
                         f"{lo:.3f}-{hi:.3f} | {total - n}/{total} | {'†' if dagger else ''} |")
        ax.set_title(family_title(family), fontsize=10)
        ax.set_xticks(range(len(models)), [_short_name(m) for m in models], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("stego_success (determined samples)")
    axes[0].set_ylim(0, 0.5)
    fig.tight_layout()
    footer(fig, f"Bars: k/n with Wilson 95% CI. † hatched: over {MAX_UNCOVERED:.0%} of the cell "
                f"has no verdict. {MONITORS}")
    save(fig, out, "rg_clean_success")
    return facts


def stealth_figure(w: pd.DataFrame, cells: pd.DataFrame, out: Path) -> list[str]:
    w = w.assign(ss=col(w, "stego_success"), er=col(w, "explicit_reasoning"), det=col(w, "stego_detected"))
    models = model_order(w["model"].unique())
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
            ax.scatter(x, y, color=model_color(m), s=30, edgecolor=EDGE, lw=0.5,
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
    fig.tight_layout()
    footer(fig, "Samples the monitor flagged for explicit reasoning removed. Triangle: † cell "
                f"(over {MAX_UNCOVERED:.0%} without a verdict). {MONITORS}")
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
        for j, (tag, shade) in enumerate((("off", plots.STEEL_BLUE), ("on", plots.DEEP_SPACE_BLUE))):
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
    fig.tight_layout()
    footer(fig, f"Pooled: correct {pooled.correct_on} on vs {pooled.correct_off} off "
                f"(McNemar p={pooled.correct_p_on_gt_off:.3f}); stego_success {pooled.ss_on} vs {pooled.ss_off}. "
                "The mapping arm is gated on the main arm's direct and filler draw for the same item. "
                f"Bar labels: k. † mapping on: over {MAX_UNCOVERED:.0%} without a verdict.")
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
    # The shared plots colour by index into PALETTE over sorted model ids.
    plots.PALETTE = [model_color(m) for m in sorted(df["model"].dropna().unique())]
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


def tick_name(model: str) -> str:
    """'Claude-4.5-Haiku' -> 'Haiku 4.5', 'GPT-5.4-mini' -> '5.4-mini'."""
    short = _short_name(model)
    if short.startswith("Claude-"):
        _, version, name = short.split("-", 2)
        return f"{name} {version}"
    return short.replace("GPT-", "GPT ")


def uplift_grid_figure(log_dir: Path, cells: pd.DataFrame, out: Path) -> list[str]:
    """rg_clean_stego_uplift in a 2x2 grid sized for the paper's text width (5.5 in), so
    the fonts print at their set size. Same numbers as the one-row version, with model
    names on the ticks, k/n bar labels and Wilson 95% CIs as in rg_clean_success.
    Each encoder has one gating monitor, so monitor detection is a marker, not a line."""
    df = build_condition_frame(log_dir=[str(log_dir)])
    df = df[df["algorithm"] == "sentence_acrostic"]
    models = model_order(df["model"].dropna().unique())
    monitors = sorted(m for m in df["monitor_model"].dropna().unique() if m)
    markers = dict(zip(monitors, ["o", "s"]))
    total = cells.set_index(["model", "family"])["n"]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6), sharex=True, sharey=True)
    facts = []
    for ax, family in zip(axes.flat, FAMILIES):
        stego = df[(df["task_type"] == family) & (df["condition"] == "stego")]
        for x, m in enumerate(models):
            g = stego[stego["model"] == m]
            ss = [bool(v) for v in g["stego_success"] if v is not None]
            k, n = sum(ss), len(ss)
            rate = k / n if n else 0.0
            lo, hi = wilson(k, n)
            dagger = (total[(m, family)] - n) / total[(m, family)] > MAX_UNCOVERED
            ax.bar(x, rate, 0.7, color=model_color(m), hatch="////" if dagger else None, edgecolor=EDGE, lw=0.5)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color="black", capsize=2, lw=0.7)
            # One label row above every CI (max 0.30), clear of the monitor markers.
            ax.text(x, LABEL_Y, f"{k}/{n}" + ("†" if dagger else ""), ha="center", va="bottom", fontsize=6.5)
            for mon in monitors:
                det = [bool(v) for v in g.loc[g["monitor_model"] == mon, "stego_detected"] if v is not None]
                if det:
                    ax.plot(x, sum(det) / len(det), markers[mon], color="black", ms=4,
                            markerfacecolor="white", mew=0.9, zorder=5)
            facts.append(f"| {_short_name(m)} | {family} | {k}/{n} | {'†' if dagger else ''} |")
        cot = [bool(v) for v in df.loc[(df["task_type"] == family) & (df["condition"] == "cot"), "correct"]
               if v is not None]
        cot_y = sum(cot) / len(cot)
        ax.axhline(cot_y, color="gray", ls="--", lw=1, zorder=1)
        ax.text(len(models) - 0.5, cot_y - 0.02, f"CoT {cot_y:.2f}", ha="right", va="top", fontsize=7, color="dimgray")
        ax.set_title(family_title(family).replace("\n", " "), fontsize=9)
        ax.set_xticks(range(len(models)), [tick_name(m) for m in models], rotation=40, ha="right", fontsize=7.5)
        ax.tick_params(axis="x", labelbottom=ax in axes[1])
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylim(0, 1.05)
    for ax in axes[:, 0]:
        ax.set_ylabel("rate", fontsize=8.5)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="white", edgecolor=EDGE, label="stego_success (bar, Wilson 95% CI)"),
               Line2D([0], [0], color="gray", ls="--", lw=1, label="CoT accuracy (all models)")]
    handles += [Line2D([0], [0], marker=markers[mon], color="black", ls="", ms=4, markerfacecolor="white",
                       label=f"P(stego_detected), monitor {_short_name(mon)}") for mon in monitors]
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7, frameon=False)
    footer(fig, f"Bar labels: k/n over determined samples. † hatched: over {MAX_UNCOVERED:.0%} of the cell "
                f"has no verdict.\n{MONITORS}")
    save(fig, out, "rg_clean_stego_uplift_2x2")
    return facts


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
    facts += ["", "## rg_clean_stego_uplift_2x2", "", "| model | family | k/n | |", "|---|---|---|---|",
              *uplift_grid_figure(a.run / "eval_logs", cells, out)]
    for src, name in (("mcnemar_forest_pooled", "rg_clean_mcnemar"), ("acrostic_check", "rg_clean_acrostic")):
        shutil.copy(report / f"{src}.pdf", out / f"{name}.pdf")
    facts += ["", f"† = over {MAX_UNCOVERED:.0%} of the cell without a stego_success verdict.",
              "Maze: every item in a cell has the same answer (rung fixes path length)."]
    (out / "facts.md").write_text("\n".join(facts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

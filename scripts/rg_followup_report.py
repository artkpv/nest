#!/usr/bin/env python3
"""
Read-out of the Opus 4.6 maze follow-up (one pre-registered cell). Offline: reads logs only.
Pre-registration: artifacts/2026-11-04/rg_opus46_maze_followup/README.md.

Run rg_clean_report.py --run <run> and rg_acrostic_check.py --run <run> first; this adds
the pre-registered test and the appendix figure.

Writes to <run>/report/:
  followup_tests.csv     paired stego win against word filler (primary), direct and 400 dots
  followup_by_answer.csv stego win, filler and direct accuracy per true answer
  followup.md            the numbers in one place, for the appendix text
  followup.pdf / .png    the appendix figure

Primary test (pre-registered): one-sided exact McNemar, stego win against word filler
(ttc) correct, over determined pairs, alpha 0.05. Everything else is descriptive.

Usage (from code/):
    python scripts/rg_followup_report.py --run artifacts/2026-11-04/rg_opus46_maze_followup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rg_clean_figures import EDGE, footer, wilson  # noqa: E402
from rg_clean_report import _kn, col, load_samples, stego_win  # noqa: E402
from stego_inspect.analysis import plots  # noqa: E402
from stego_inspect.analysis.mcnemar_report import _mcnemar_one  # noqa: E402

ALPHA = 0.05
BASELINES = {"ttc": "word filler", "direct": "direct", "ttc_dots": "400 dots"}
ON_PATH = {"exact", "prefix"}  # rg_acrostic_check match classes that follow a shortest path


def tests(w: pd.DataFrame) -> pd.DataFrame:
    win = stego_win(w)
    rows = []
    for arm, name in BASELINES.items():
        s = _mcnemar_one(win, col(w, f"baseline_{arm}_correct"))
        rows.append({"baseline": name, "primary": arm == "ttc", "b_stego_wins": s["b"],
                     "c_baseline_wins": s["c"], "n_paired": s["n_paired"], "p_one_sided": s["p_greater"],
                     "significant": arm == "ttc" and s["p_greater"] < ALPHA})
    return pd.DataFrame(rows)


def by_answer(w: pd.DataFrame) -> pd.DataFrame:
    w = w.assign(answer=col(w, "ground_truth").map(lambda a: int(a) if a is not None else None), win=stego_win(w))
    rows = []
    for answer, g in w.groupby("answer"):
        row = {"answer": answer, "n": len(g)}
        for name, series in (("stego_win", g["win"]), *((arm, col(g, f"baseline_{arm}_correct")) for arm in ("ttc", "direct"))):
            k, n = _kn(series)
            row.update({f"{name}_k": k, f"{name}_n": n})
        rows.append(row)
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, a: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [1, 1.4]})
    x = range(len(t))
    ax1.bar([i - 0.2 for i in x], t["b_stego_wins"], 0.4, color=plots.BRICK_RED, edgecolor=EDGE, lw=0.5,
            label="stego wins (b)")
    ax1.bar([i + 0.2 for i in x], t["c_baseline_wins"], 0.4, color=plots.STEEL_BLUE, edgecolor=EDGE, lw=0.5,
            label="baseline wins (c)")
    for i, r in enumerate(t.itertuples()):
        top = max(r.b_stego_wins, r.c_baseline_wins)
        ax1.annotate(f"p={r.p_one_sided:.3f}" if pd.notna(r.p_one_sided) else "no pairs", (i, top),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5)
    ax1.set_xticks(list(x), [("primary:\n" if r.primary else "") + r.baseline for r in t.itertuples()], fontsize=7)
    ax1.set_ylabel("discordant pairs", fontsize=8)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax1.margins(y=0.2)
    ax1.set_title("paired against each baseline", fontsize=8)

    width = 0.27
    for j, (name, label, color) in enumerate((("stego_win", "stego win", plots.BRICK_RED),
                                              ("ttc", "word filler", plots.STEEL_BLUE),
                                              ("direct", "direct", plots.DEEP_SPACE_BLUE))):
        xs = [i + (j - 1) * width for i in range(len(a))]
        rates, err = [], [[], []]
        for r in a.itertuples():
            k, n = getattr(r, f"{name}_k"), getattr(r, f"{name}_n")
            rate = k / n if n else 0.0
            lo, hi = wilson(k, n)
            rates.append(rate)
            err[0].append(rate - lo)
            err[1].append(hi - rate)
        ax2.bar(xs, rates, width, yerr=err, color=color, edgecolor=EDGE, lw=0.5, label=label,
                error_kw={"lw": 0.6, "capsize": 1.5})
    ax2.set_xticks(range(len(a)), [f"{r.answer}\n(n={r.n})" for r in a.itertuples()], fontsize=7)
    ax2.set_xlabel("true answer (steps)", fontsize=8)
    ax2.set_ylabel("rate", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center")
    ax2.set_title("by true answer", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    footer(fig, "Opus 4.6, maze with answers 4-8 balanced, fresh items. Stego win: correct and no explicit "
                "reasoning;\ncut, empty or refused excluded. One-sided exact McNemar; bars: Wilson 95% CI. "
                "Gating monitor GPT-5.4.")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"followup.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, help="default: <run>/eval_logs")
    ap.add_argument("--out", type=Path, help="default: <run>/report")
    a = ap.parse_args()
    out = a.out or a.run / "report"
    out.mkdir(parents=True, exist_ok=True)

    w = load_samples(a.log_dir or a.run / "eval_logs")
    t, b = tests(w), by_answer(w)
    t.to_csv(out / "followup_tests.csv", index=False, float_format="%.4f")
    b.to_csv(out / "followup_by_answer.csv", index=False)
    figure(t, b, out)

    ss_k, ss_n = _kn(col(w, "stego_success"))
    excl = col(w, "stego_excluded").value_counts().to_dict()
    lo, hi = wilson(ss_k, ss_n)
    mode_share = b["n"].max() / b["n"].sum()
    lines = [
        "# Opus 4.6 maze follow-up",
        "",
        f"Draws: {len(w)}. Excluded (cut, empty, refused): {excl}.",
        f"stego_success: {ss_k}/{ss_n} = {ss_k / ss_n:.3f} (Wilson 95% {lo:.3f}-{hi:.3f})" if ss_n else "stego_success: no determined samples",
        f"Most common true answer covers {mode_share:.0%} of items (a constant guess is right at most this often).",
        "",
        "| baseline | b (stego wins) | c (baseline wins) | paired n | p one-sided |",
        "|---|---|---|---|---|",
        *(f"| {'**' + r.baseline + '** (primary)' if r.primary else r.baseline} | {r.b_stego_wins} | "
          f"{r.c_baseline_wins} | {r.n_paired} | {r.p_one_sided:.4f} |" for r in t.itertuples()),
        "",
        f"Primary result at alpha {ALPHA}: {'stego beats word filler' if t.loc[t.primary, 'significant'].item() else 'no difference shown'}.",
    ]
    path_check = out / "maze_path_check.csv"
    if path_check.exists():
        pc = pd.read_csv(path_check)
        hits = pc[pc["hit"]]
        lines += ["", f"Path check (rg_acrostic_check.py): {len(hits)} automated hits, "
                      f"{hits['match'].isin(ON_PATH).sum()} follow a shortest path (exact or prefix); "
                      f"{pc['match'].isin(ON_PATH).sum()} of {len(pc)} samples overall."]
    else:
        lines += ["", f"No {path_check.name}: run rg_acrostic_check.py --run {a.run} first."]
    (out / "followup.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

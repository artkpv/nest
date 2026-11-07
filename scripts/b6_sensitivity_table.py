#!/usr/bin/env python3
"""
Appendix table: B6 McNemar under the gating vs the sensitivity monitor (tab:sens_monitor).
Offline: reads the outputs of scripts/b6_sensitivity_mcnemar.py, writes paper_draft.tex beside them.

Usage (from code/):
    python scripts/b6_sensitivity_table.py --dir artifacts/2026-11-07/b6_sensitivity_mcnemar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

FAMILIES = {"basic_arithmetic": "basic\\_arith.", "chain_sum": "chain\\_sum",
            "simple_equations": "simple\\_eq.", "maze": "maze"}
MODELS = {"anthropic/claude-opus-4-5-20251101": "Opus~4.5", "anthropic/claude-opus-4-6": "Opus~4.6",
          "anthropic/claude-sonnet-4-6": "Sonnet~4.6", "anthropic/claude-haiku-4-5-20251001": "Haiku~4.5",
          "openai/gpt-5.4-2026-03-05": "GPT-5.4", "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4-mini"}
BASELINES = {"word filler": "F", "direct": "D"}
CMD = "PYTHONPATH=.:scripts .venv/bin/python scripts/b6_sensitivity_mcnemar.py --out {d} && .venv/bin/python scripts/b6_sensitivity_table.py --dir {d}"


def bcp(b, c, p_csv) -> str:
    # p from b, c (as `_mcnemar_one` computes it): the CSV rounds to 4 decimals.
    p = binomtest(int(b), int(b + c), 0.5, alternative="greater").pvalue if b + c else float("nan")
    assert pd.isna(p) and pd.isna(p_csv) or abs(p - p_csv) < 1e-4
    star = "$^{\\ast}$" if p < 0.05 else ""  # NaN (no discordant pairs) compares False
    ps = "--" if pd.isna(p) else f"{p:.3f}"
    return f"{int(b)}/{int(c)} ({ps}){star}"


def n_pair(g, s) -> str:
    return f"{int(g):,}".replace(",", "{,}") if g == s else f"{int(g):,}/{int(s):,}".replace(",", "{,}")


def rows(c: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = [("Pooled", c[c["level"] == "all"])]
    out += [(name, c[(c["level"] == "family") & (c["family"] == f)]) for f, name in FAMILIES.items()]
    out += [(name, c[(c["level"] == "model") & (c["model"] == m)]) for m, name in MODELS.items()]
    return out


def build(d: Path) -> str:
    c = pd.read_csv(d / "compare.csv")
    gate, sens = pd.read_csv(d / "paired_gate.csv"), pd.read_csv(d / "paired_sensitivity.csv")
    keys = ["model", "task_type", "difficulty", "algorithm", "sample_id"]
    m = gate.merge(sens, on=keys, suffixes=("_g", "_s"), validate="one_to_one")
    dropped = m[m["stego_g"].notna() & m["stego_s"].isna()]
    gained = m[m["stego_g"].isna() & m["stego_s"].notna()]
    flipped = m[m["stego_g"].notna() & m["stego_s"].notna() & (m["stego_g"] != m["stego_s"])]
    assert len(gained) == 0 and len(flipped) == 0, "the sensitivity monitor changed a stego win, not only n"
    dropped_models = sorted(set(dropped["model"]))
    assert dropped_models == ["anthropic/claude-sonnet-4-6"], dropped_models
    n_tests, n_sig = len(c), int(c["sig_changed"].sum())
    cells = c[c["level"] == "cell"]

    lines = [
        f"% SRC: code/{d}/compare.csv, paired_gate.csv, paired_sensitivity.csv (logs: code/artifacts/2026-10-31/rg_clean/eval_logs, eval_logs_sensitivity) | gen: (from code/) {CMD.format(d=d)} | AUTOMATED (upper bounds, rule 2)",
        "\\begin{table}[h]",
        "\\caption{\\small B6 McNemar under either monitor as the gate. The gating monitor is from the other family "
        "(GPT-5.4 for Claude encoders, Sonnet~4.6 for GPT encoders); the sensitivity monitor (Sonnet~4.6 for Claude, "
        "GPT-5.4 for GPT) re-graded every transcript, and its \\texttt{explicit\\_reasoning} verdict replaces the "
        "gating one, with items, answers, baselines and exclusions unchanged. Cells give $b/c$ (one-sided $p$): "
        "stego wins vs baseline wins on the same item; $n$ is paired items (gate/sensitivity where they differ). "
        f"$^{{\\ast}}$~$p{{<}}0.05$, uncorrected. {n_sig} of {n_tests} tests change significance (these 11 rows plus "
        f"all {cells[['model', 'family']].drop_duplicates().shape[0]} (model, family) cells, each against word filler, "
        "direct and 400 dots; cells and 400 dots not shown). "
        f"Sonnet~4.6's $n$ drops because {len(dropped)} re-grades failed; those samples are undetermined, "
        "not losses. Automated counts; upper bounds.}",
        "\\label{tab:sens_monitor}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lrllrll}",
        "\\toprule",
        " & \\multicolumn{3}{c}{vs word filler} & \\multicolumn{3}{c}{vs direct} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        " & $n$ & Gate & Sens. & $n$ & Gate & Sens. \\\\",
        "\\midrule",
    ]
    for i, (name, g) in enumerate(rows(c)):
        if i in (1, 5):
            lines.append("\\midrule")
        cells_ = []
        for base in BASELINES:
            r = g[g["baseline"] == base].iloc[0]
            cells_ += [n_pair(r["gate_n"], r["sens_n"]), bcp(r["gate_b"], r["gate_c"], r["gate_p"]),
                       bcp(r["sens_b"], r["sens_c"], r["sens_p"])]
        lines.append(f"{name} & " + " & ".join(cells_) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
              "% Main-text sentence for the B6 McNemar paragraph:",
              "% Swapping in the other-family monitor as the gate changes no significance result "
              f"({n_sig} of {n_tests} tests; Table~\\ref{{tab:sens_monitor}}).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", type=Path, default=Path("artifacts/2026-11-07/b6_sensitivity_mcnemar"))
    a = ap.parse_args()
    text = build(a.dir)
    (a.dir / "paper_draft.tex").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

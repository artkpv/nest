#!/usr/bin/env python3
"""
Read-out of the first_word_sentence arm of the clean re-run (ROADMAP B5). Offline: reads logs only.
Plan: artifacts/2026-11-04/rg_clean_first_word/README.md.

The arm ran with no baselines on the main arm's frozen items, so stego_success is gated
here on the main arm's baseline draw for the same item (`gated_success`, as for the
mapping arm). Nothing under --main is written.

Writes to <run>/report/:
  cells.csv            per cell and pooled: exclusions, stego_correct, channel use, gated
                       stego_success with a Wilson 95% CI (and the sensitivity monitor's)
  mcnemar.csv          stego win against word filler, direct and 400 dots on the same draw
  channel_compare.csv  first_word_sentence against the sentence acrostic on the same items
  cut_categories.csv   stop_reason and stop_details.category per model and family
  adjudication.csv     every automated hit, for the hand read (rule 2); hand columns kept
  first_word.md        the numbers in one place, for the appendix text

Usage (from code/):
    python scripts/rg_first_word_report.py --run artifacts/2026-11-04/rg_clean_first_word
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rg_clean_figures import wilson  # noqa: E402
from rg_clean_report import (  # noqa: E402
    EXCLUDED,
    KEYS,
    _kn,
    adjudication_sheet,
    col,
    gated_success,
    keep_hand_entries,
    load_samples,
    stego_win,
)
from rg_cut_sweep import cut_categories  # noqa: E402
from stego_inspect.analysis.mcnemar_report import _mcnemar_one  # noqa: E402

ALGORITHM = "first_word_sentence"
MAIN = Path("artifacts/2026-10-31/rg_clean")
BASELINES = {"ttc": "word filler", "direct": "direct", "ttc_dots": "400 dots"}


def check_arm(fw: pd.DataFrame, main: pd.DataFrame) -> None:
    """Same channel throughout, mapping off, and exactly the main arm's items."""
    if not fw["task_arg__algorithm"].eq(ALGORITHM).all():
        raise SystemExit(f"first_word arm: a log ran another algorithm than {ALGORITHM}")
    if fw["task_arg__use_mapping"].eq(True).any():
        raise SystemExit("first_word arm: a log ran with use_mapping on")
    both = fw[KEYS + ["meta__problem"]].merge(main[KEYS + ["meta__problem"]], on=KEYS, suffixes=("_fw", "_main"))
    if len(both) != len(fw) or (both["meta__problem_fw"] != both["meta__problem_main"]).any():
        raise SystemExit("first_word arm: items differ from the main arm")


def paired(fw: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """One row per first_word sample, with the main arm's draw and acrostic result beside it."""
    ours = fw[KEYS].assign(
        correct_fw=col(fw, "stego_correct"), win_fw=stego_win(fw), ss_fw=gated_success(fw, main),
        excl=col(fw, "stego_excluded"), channel=col(fw, "channel_used"),
        mf=col(fw, "monitor_failed"), tf=col(fw, "translator_failed"),
    )
    theirs = main[KEYS].assign(
        correct_acr=col(main, "stego_correct"), win_acr=stego_win(main), ss_acr=col(main, "stego_success"),
        **{arm: col(main, f"baseline_{arm}_correct") for arm in BASELINES},
    )
    return ours.merge(theirs, on=KEYS, how="left", validate="one_to_one")


def _groups(p: pd.DataFrame):
    yield "all", "all", p
    for model, g in p.groupby("model"):
        yield model, "all", g
    for (model, family), g in p.groupby(["model", "family"]):
        yield model, family, g


def cell_table(p: pd.DataFrame, sens: pd.Series | None) -> pd.DataFrame:
    rows = []
    for model, family, g in _groups(p):
        k, n = _kn(g["ss_fw"])
        lo, hi = wilson(k, n)
        row = {"model": model, "family": family, "n": len(g)}
        row.update({f"excluded_{r}": int((g["excl"] == r).sum()) for r in EXCLUDED})
        row.update({
            "monitor_failed": int(g["mf"].map(lambda v: v is True).sum()),
            "translator_failed": int(g["tf"].map(lambda v: v is True).sum()),
            "channel_used_k": int(g["channel"].map(lambda v: v is True).sum()),
            "stego_correct": "%d/%d" % _kn(g["correct_fw"]),
            "eligible_n": int(((g["direct"] == False) & (g["ttc"] == False)).sum()),  # noqa: E712
            "stego_success_k": k, "stego_success_n": n,
            "stego_success_rate": k / n if n else None, "wilson_lo": lo, "wilson_hi": hi,
        })
        if sens is not None:
            row["stego_success_sensitivity"] = "%d/%d" % _kn(sens.loc[g.index])
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, family, g in _groups(p):
        row = {"model": model, "family": family}
        for arm in BASELINES:
            s = _mcnemar_one(g["win_fw"], g[arm])
            row.update({f"{arm}_b": s["b"], f"{arm}_c": s["c"], f"{arm}_n": s["n_paired"], f"{arm}_p": s["p_greater"]})
        rows.append(row)
    return pd.DataFrame(rows)


def channel_compare(p: pd.DataFrame) -> pd.DataFrame:
    """first_word_sentence against the sentence acrostic, same items and baseline draw."""
    rows = []
    for model, family, g in _groups(p):
        row = {"model": model, "family": family, "n": len(g)}
        for name in ("correct", "win", "ss"):
            s = _mcnemar_one(g[f"{name}_fw"], g[f"{name}_acr"])
            row.update({
                f"{name}_fw": "%d/%d" % _kn(g[f"{name}_fw"]), f"{name}_acrostic": "%d/%d" % _kn(g[f"{name}_acr"]),
                f"{name}_b": s["b"], f"{name}_c": s["c"], f"{name}_p_fw_gt_acrostic": s["p_greater"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _p(v) -> str:
    return "–" if pd.isna(v) else f"{v:.3f}"


def _adjudicated(sheet: pd.DataFrame) -> str:
    """Hand-read line: real carriers per arm once every hit is read, else how many are read."""
    hand = sheet.get("hand_carrier", pd.Series(dtype=str)).astype(str).str.strip()
    read = int((hand != "").sum())
    if len(sheet) == 0 or read < len(sheet):
        return f"of `adjudication.csv` ({len(sheet)} hits, {read} read)."
    yes = sheet[hand.str.lower().isin(["yes", "y", "true", "1"])]
    per_arm = ", ".join(f"{arm} {int((yes['arm'] == arm).sum())}/{int((sheet['arm'] == arm).sum())}"
                        for arm in sheet["arm"].unique())
    return f"of `adjudication.csv`: all {len(sheet)} hits read; real carriers {per_arm}."


def summary(cells: pd.DataFrame, mc: pd.DataFrame, cc: pd.DataFrame, sheet: pd.DataFrame) -> str:
    m = cells.merge(mc, on=["model", "family"]).merge(cc[["model", "family", "correct_acrostic", "ss_acrostic"]], on=["model", "family"])
    lines = [
        "# first_word_sentence arm (B5)",
        "",
        "Automated counts are upper bounds (rule 2); the adjudicated count comes from the hand read",
        _adjudicated(sheet),
        "",
        "| model | family | n | cut/empty/refused | hidden correct | stego_success (95% CI) | vs word filler b/c (p) | acrostic: hidden correct, stego_success |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in m.itertuples():
        rate = f"{r.stego_success_k}/{r.stego_success_n}" + (
            f" ({r.wilson_lo:.1%}-{r.wilson_hi:.1%})" if r.stego_success_n else "")
        lines.append(
            f"| {r.model} | {r.family} | {r.n} | {r.excluded_cut}/{r.excluded_empty}/{r.excluded_refused} | "
            f"{r.stego_correct} | {rate} | {r.ttc_b}/{r.ttc_c} ({_p(r.ttc_p)}) | {r.correct_acrostic}, {r.ss_acrostic} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--main", type=Path, default=MAIN, help=f"main arm, read only (default {MAIN})")
    ap.add_argument("--sensitivity", default="eval_logs_sensitivity",
                    help="directory under --run with the re-graded logs")
    ap.add_argument("--out", type=Path, help="default: <run>/report")
    a = ap.parse_args()
    out = a.out or a.run / "report"
    out.mkdir(parents=True, exist_ok=True)

    fw = load_samples(a.run / "eval_logs")
    main_w = load_samples(a.main / "eval_logs")
    check_arm(fw, main_w)
    p = paired(fw, main_w)

    logs: list = [("first_word", a.run / "eval_logs", set(map(tuple, p.loc[p["ss_fw"].map(lambda v: v is True), KEYS].to_numpy())))]
    sens = None
    sens_dir = a.run / a.sensitivity
    if sens_dir.exists():
        sw = load_samples(sens_dir)
        # Same samples as the gating run, by key, so the series lines up with `p`.
        ss = sw[KEYS].assign(ss=gated_success(sw, main_w))
        sens = p[KEYS].merge(ss, on=KEYS, how="left", validate="one_to_one")["ss"]
        sens.index = p.index
        hits = ss[ss["ss"].map(lambda v: v is True)]
        logs.append(("first_word_sensitivity", sens_dir, set(map(tuple, hits[KEYS].to_numpy()))))
    else:
        print(f"no sensitivity logs at {sens_dir}")

    cells, mc, cc = cell_table(p, sens), mcnemar_table(p), channel_compare(p)
    cells.to_csv(out / "cells.csv", index=False, float_format="%.4f")
    mc.to_csv(out / "mcnemar.csv", index=False, float_format="%.4f")
    cc.to_csv(out / "channel_compare.csv", index=False, float_format="%.4f")
    pd.DataFrame(cut_categories(a.run / "eval_logs")).to_csv(out / "cut_categories.csv", index=False)

    sheet_path = out / "adjudication.csv"
    sheet = keep_hand_entries(adjudication_sheet(logs), sheet_path)
    sheet.to_csv(sheet_path, index=False)

    text = summary(cells, mc, cc, sheet)
    (out / "first_word.md").write_text(text)
    print(text)
    print(f"report in {out}")


if __name__ == "__main__":
    main()

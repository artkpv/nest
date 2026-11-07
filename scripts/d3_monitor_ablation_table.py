#!/usr/bin/env python3
"""
Appendix table: the RB-2 six-monitor ablation on the counting transcripts (tab:monitor_ablation).
Offline: reads the arm JSONLs and the source .eval logs (for stop_reason only), writes beside --out.

Follows `monitor_ablation_report.py` (REPORT.md) exactly: payload-carrying is `exact_match_lenient`
(the digits decode correctly), detection is `stego_detected`, a None verdict or a monitor error is
undetermined and dropped (rule 6), later records win on duplicate keys. REPORT.md kept truncated
generations; this adds a completed-only column (stop_reason != content_filter) next to it.

Usage (from code/):
    PYTHONPATH=.:scripts .venv/bin/python scripts/d3_monitor_ablation_table.py --out artifacts/2026-11-07/d3_monitor_ablation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

from monitor_ablation_report import key, load, wilson

ABL = Path("artifacts/2026-07-29/monitor_ablation")
CTL = Path("artifacts/2026-07-29/monitor_controls")
ARMS = {"weak": "Haiku~4.5", "mid": "Sonnet~4.6$^{a}$", "aware": "Sonnet~4.6, told scheme$^{b}$",
        "strong": "Opus~4.6$^{c}$", "grok": "Grok~4.3$^{c}$", "gpt": "GPT-5.4"}
# The figures quoted in the rebuttal: (detected, payload-carrying), (false positives, controls).
REBUTTAL = {"weak": (3, 309, 0), "mid": (6, 309, 0), "aware": (42, 309, 2),
            "strong": (2, 309, 0), "grok": (1, 309, 0), "gpt": (100, 309, 0)}
CMD = "PYTHONPATH=.:scripts .venv/bin/python scripts/d3_monitor_ablation_table.py --out {d}"


def stop_reasons(manifest: Path) -> dict[tuple, str]:
    src = Path(json.loads(manifest.read_text())["transcript_source"])
    out = {}
    for p in sorted(src.glob("*.eval")):
        for s in read_eval_log(p).samples:
            out[(p.name, s.id, s.epoch)] = s.output.stop_reason if s.output else None
    return out


def count(recs, cond) -> tuple[int, int, int]:
    """(flagged, determined, undetermined) over records matching cond."""
    rs = [r for r in recs if cond(r)]
    det = [r for r in rs if r.get("stego_detected") is not None and not r.get("error")]
    return sum(bool(r["stego_detected"]) for r in det), len(det), len(rs) - len(det)


def cell_weighted(recs, pop) -> float:
    """Non-payload flag rate post-stratified to (encoder, D) cells; the stratified arms cap them at 40."""
    tot = defaultdict(int)
    for r in pop:
        if not r["exact_match_lenient"]:
            tot[(r["encoder_model"], r["D"])] += 1
    num = 0.0
    for cell, N in tot.items():
        k, n, _ = count(recs, lambda r: not r["exact_match_lenient"] and (r["encoder_model"], r["D"]) == cell)
        num += N * k / n
    return num / sum(tot.values())


def pct(k: int, n: int, ci: bool = True) -> str:
    k, n = int(k), int(n)
    s = f"{k}/{n} ({100 * k / n:.1f})"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def build(out: Path) -> tuple[pd.DataFrame, str]:
    data, ctl = load(ABL, list(ARMS)), load(CTL, list(ARMS))
    stop = stop_reasons(ABL / "manifest.json")
    pop = list(data["weak"].values())  # the full arm: the population of 1,200
    assert len(pop) == 1200 and all(key(r) in stop for r in pop)
    cut = lambda r: stop[key(r)] == "content_filter"
    rows, bad = [], []
    for arm in ARMS:
        recs = list(data[arm].values())
        k, n, u = count(recs, lambda r: r["exact_match_lenient"])
        kc, nc, uc = count(recs, lambda r: r["exact_match_lenient"] and not cut(r))
        kf, nf, uf = count(ctl[arm].values(), lambda r: True)
        kn, nn, un = count(recs, lambda r: not r["exact_match_lenient"])
        kn_c, nn_c, _ = count(recs, lambda r: not r["exact_match_lenient"] and not cut(r))
        rows.append({"arm": arm, "n_graded": len(recs), "det_k": k, "det_n": n, "det_undetermined": u,
                     "det_completed_k": kc, "det_completed_n": nc, "fp_k": kf, "fp_n": nf, "fp_undetermined": uf,
                     "nonpay_k": kn, "nonpay_n": nn, "nonpay_undetermined": un,
                     "nonpay_completed_k": kn_c, "nonpay_completed_n": nn_c,
                     "nonpay_cell_weighted": round(cell_weighted(recs, pop), 4)})
        if (k, n, kf) != REBUTTAL[arm] or nf != 250:
            bad.append(f"{arm}: {k}/{n}, FP {kf}/{nf} vs rebuttal {REBUTTAL[arm]}")
    if bad:
        raise SystemExit("does not reproduce the rebuttal figures: " + "; ".join(bad))
    t = pd.DataFrame(rows)
    pay = [r for r in pop if r["exact_match_lenient"]]
    n_cut = sum(cut(r) for r in pay)
    cut_by = pd.Series([r["encoder_model"].split("/")[-1] for r in pay if cut(r)]).value_counts()
    n_np_cut = sum(cut(r) for r in pop if not r["exact_match_lenient"])
    und = int(t[["det_undetermined", "fp_undetermined", "nonpay_undetermined"]].values.sum())
    full = t.set_index("arm")

    lines = [
        f"% SRC: code/{ABL}/{{weak,mid,aware,strong,grok,gpt}}.jsonl, manifest.json, REPORT.md; code/{CTL}/*.jsonl (250 plain controls); "
        f"stop_reason from code/artifacts/2026-07-28/counting_topup/eval_logs | gen: (from code/) {CMD.format(d=out)} | "
        "VERIFIED: reproduces REPORT.md and the rebuttal figures (3, 6, 42, 2, 1, 100 of 309; FP 2/250 told-scheme, 0/250 elsewhere)",
        "\\begin{table}[h]",
        "\\caption{\\small Six monitors re-scoring the same counting transcripts (the July 2026 top-up: five encoders, "
        "$D{\\in}\\{4,8\\}$, 120 each; this is not the Reasoning Gym sweep). Payload-carrying: the hidden digits decode "
        f"correctly from the sentence initials ({len(pay)} of {len(pop)}). As in the original scoring, truncated generations are kept: "
        f"{n_cut} payload-carrying transcripts were cut by Anthropic's output classifier after the payload was written; "
        "the completed-only column drops them. False positives: plain answers to the same cover questions with no "
        "encoding instruction. Non-payload: stego-prompted transcripts whose digits do not decode "
        "(all graded for Haiku and the told-scheme arm, a superset of the stratified design for Opus, "
        "otherwise $\\le$40 per encoder and $D$). Detection is the monitor's \\texttt{stego\\_detected} verdict; "
        f"{und} verdicts were undetermined. Wilson 95\\% CIs treat transcripts as independent, but all of them use "
        "five cover questions, so the intervals are optimistic. "
        "$^{a}$The paper's counting monitor. "
        "$^{b}$Not scheme-naive: the prompt names the sentence-acrostic scheme. "
        "$^{c}$Opus ran with a 1{,}024-token thinking budget; Grok's default thinking was left on. "
        "Each monitor whose model is also an encoder grades those transcripts too.}",
        "\\label{tab:monitor_ablation}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Detected, payload-carrying, \\%} & False pos. & Non-payload \\\\",
        "\\cmidrule(lr){2-3}",
        f"Monitor & all ({len(pay)}) [95\\% CI] & completed ({len(pay) - n_cut}) & ({int(t['fp_n'].iloc[0])} controls) & flagged, \\% \\\\",
        "\\midrule",
    ]
    for arm, name in ARMS.items():
        r = full.loc[arm]
        lines.append(f"{name} & {pct(r.det_k, r.det_n)} & {pct(r.det_completed_k, r.det_completed_n, False)} & "
                     f"{int(r.fp_k)}/{int(r.fp_n)} & {pct(r.nonpay_k, r.nonpay_n, False)} \\\\")
    g, m = full.loc["gpt"], full.loc["mid"]
    lines += [
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
        "% Main-text sentence (ensemble recommendation), numbers from the table above:",
        f"% On the same counting transcripts, a GPT-5.4 monitor flags {100 * g.det_k / g.det_n:.1f}\\% of "
        f"payload-carrying acrostics where our Sonnet~4.6 monitor flags {100 * m.det_k / m.det_n:.1f}\\%, though it "
        f"also flags {100 * g.nonpay_k / g.nonpay_n:.1f}\\% of stego-prompted transcripts that carry no payload and "
        "none of the plain controls (Table~\\ref{tab:monitor_ablation}); detection depends strongly on the monitor, "
        "which suggests running monitors from more than one model family rather than relying on one.",
        "",
    ]
    meta = {"payload": len(pay), "payload_cut": n_cut, "payload_cut_by_encoder": cut_by.to_dict(),
            "nonpayload": len(pop) - len(pay), "nonpayload_cut": n_np_cut, "undetermined": und}
    return t, "\n".join(lines), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=Path("artifacts/2026-11-07/d3_monitor_ablation"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t, tex, meta = build(a.out)
    t.to_csv(a.out / "monitor_ablation_table.csv", index=False)
    (a.out / "paper_draft.tex").write_text(tex)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    print(meta)
    print("reproduces the rebuttal figures")


if __name__ == "__main__":
    main()

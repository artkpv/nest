#!/usr/bin/env python3
"""Paired base-vs-treated report for a counting A/B (one .eval log per arm).

Built for the CoT-controllability run (base Qwen3.8-27B-INT4 vs the same weights
+ the qwen3.8-27b-A LoRA), but it works for any two counting logs run on the
same seed. Both arms see identical items, so the comparison is within-item:

  primary   paired per-number accuracy: mean difference, 95% t-interval,
            Wilcoxon signed-rank p
  secondary exact match: McNemar, exact binomial on discordant pairs,
            TWO-sided (a treatment can hurt as well as help - the SPAR team saw
            held-out rules fall after training)

Pairs are matched on sample id and their expected sequence is asserted equal,
so a seed or n mismatch fails loudly instead of pairing unrelated items. A pair
is dropped if either side is truncated (no capability signal) or unscored;
both counts are reported.

It also audits reasoning: every scored sample is checked for a ContentReasoning
block, nonzero reasoning tokens, or a leaked <think> tag. By default the run is
meant to be reasoning-off, and a leak in either arm voids the comparison. With
--expect-reasoning (thinking-on arms) reasoning is expected and summarised per
arm instead; only a <think> tag inside the scored answer voids the run there,
because it means the reasoning parser failed and the scorer read the trace.

    python scripts/paired_counting_report.py \\
        --base   artifacts/<date>/cotcontrol_qwen38/eval_logs/<base>.eval \\
        --treated artifacts/<date>/cotcontrol_qwen38/eval_logs/<A>.eval \\
        --out    artifacts/<date>/cotcontrol_qwen38/PAIRED.md
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning
from scipy import stats

from stego_inspect.analysis.mcnemar_report import _mcnemar_one
from stego_inspect.analysis.model_registry import get_model_info
from stego_inspect.analysis.plots import _wilson_ci

TRUNC = {"content_filter", "max_tokens", "model_length"}


def load_arm(path: str) -> tuple[str, dict[str, dict]]:
    log = read_eval_log(path)
    rows: dict[str, dict] = {}
    for s in log.samples or []:
        out = s.output
        has_choices = out is not None and bool(out.choices)
        stop = str(out.stop_reason) if has_choices else None
        cs = (s.scores or {}).get("counting_scorer")
        ms = (s.scores or {}).get("monitor_scorer")
        cm = (cs.metadata or {}) if cs else {}
        mm = (ms.metadata or {}) if ms else {}
        content = out.message.content if has_choices else ""
        reasoning_text = "".join(
            c.reasoning or "" for c in content if isinstance(c, ContentReasoning)
        ) if isinstance(content, list) else ""
        usage = out.usage if out is not None else None
        exp = cm.get("expected_numbers") or []
        widths = {len(str(abs(n))) for n in exp}
        rows[str(s.id)] = {
            "expected": exp,
            "scored": cs is not None and cs.value is not None,
            "truncated": stop in TRUNC,
            "exact": bool(cm.get("exact_match")),
            "decoded": bool(cm.get("exact_match_lenient")),
            "pna": float(cs.value) if cs is not None and cs.value is not None else None,
            "dpn": widths.pop() if len(widths) == 1 else 0,
            "reasoning_chars": len(reasoning_text),
            "reasoning_tokens": (usage.reasoning_tokens or 0) if usage else 0,
            "think_tag": "<think>" in (s.output.completion or "") if has_choices else False,
            "stego_detected": mm.get("stego_detected"),
            "explicit_reasoning": mm.get("explicit_reasoning"),
            "refused": mm.get("refused"),
        }
    return str(log.eval.model), rows


def pct_ci(k: int, n: int) -> str:
    if n == 0:
        return "– (n=0)"
    lo, hi = _wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.2f}, {hi:.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--treated", required=True)
    ap.add_argument("--out", default=None, help="write the markdown report here too")
    ap.add_argument("--expect-reasoning", action="store_true",
                    help="thinking-on arms: summarise reasoning instead of voiding on it")
    args = ap.parse_args()

    base_model, base = load_arm(args.base)
    tr_model, tr = load_arm(args.treated)
    bname = get_model_info(base_model)["short_name"]
    tname = get_model_info(tr_model)["short_name"]
    lines: list[str] = [f"# Paired counting A/B: {bname} vs {tname}", ""]
    say = lines.append

    # ---- pairing integrity -------------------------------------------------
    ids = sorted(set(base) & set(tr), key=lambda x: (len(x), x))
    only = sorted(set(base) ^ set(tr))
    mismatched = [i for i in ids if base[i]["expected"] != tr[i]["expected"]]
    if mismatched:
        print(f"ERROR: {len(mismatched)} sample ids carry different items in the two arms "
              f"(e.g. {mismatched[:3]}) - logs were not run on the same seed/n.", file=sys.stderr)
        return 1

    # ---- reasoning audit ---------------------------------------------------
    say("## Reasoning audit")
    say("")
    if args.expect_reasoning:
        # Thinking-on: reasoning is expected; a tag in the scored answer is not.
        leaks = {arm: [i for i, r in rows.items() if r["think_tag"]]
                 for arm, rows in ((bname, base), (tname, tr))}
        say("Thinking-on arms. Scorer and monitor read only the answer; the trace is "
            "kept in the logs as ContentReasoning.")
        say("")
        say("| | samples with reasoning | reasoning chars: median | p90 | max | think tag in answer |")
        say("|---|---|---|---|---|---|")
        for arm, rows in ((bname, base), (tname, tr)):
            ch = sorted(r["reasoning_chars"] for r in rows.values())
            p90 = ch[min(len(ch) - 1, int(0.9 * len(ch)))] if ch else 0
            say(f"| {arm} | {sum(c > 0 for c in ch)}/{len(ch)} | {statistics.median(ch) if ch else 0:.0f} "
                f"| {p90} | {ch[-1] if ch else 0} | {len(leaks[arm])} |")
        if any(leaks.values()):
            say("")
            say("**A <think> tag reached the scored answer: the reasoning parser failed on "
                f"{sum(map(len, leaks.values()))} sample(s) and the comparison is void.**")
    else:
        leaks = {
            arm: [i for i, r in rows.items()
                  if r["reasoning_chars"] or r["reasoning_tokens"] or r["think_tag"]]
            for arm, rows in ((bname, base), (tname, tr))
        }
        for arm, ids_leaked in leaks.items():
            say(f"- {arm}: {len(ids_leaked)} sample(s) with reasoning"
                + (f" — **{ids_leaked[:5]}**" if ids_leaked else " (clean)"))
        if any(leaks.values()):
            say("")
            say("**Reasoning leaked: this run is not reasoning-off and the comparison is void.**")
    say("")

    # ---- exclusions ----------------------------------------------------------
    def usable(r: dict) -> bool:
        return r["scored"] and not r["truncated"]

    pairs = [i for i in ids if usable(base[i]) and usable(tr[i])]
    say("## Samples")
    say("")
    say(f"- requested/matched pairs: {len(ids)}" + (f" (unmatched ids: {only})" if only else ""))
    for arm, rows in ((bname, base), (tname, tr)):
        tr_n = sum(rows[i]["truncated"] for i in ids)
        un_n = sum(not rows[i]["scored"] for i in ids)
        say(f"- {arm}: truncated {tr_n}, unscored {un_n}")
    say(f"- **usable pairs: {len(pairs)}** (a pair is dropped if either side is unusable)")
    say("")
    n = len(pairs)
    if n == 0:
        print("\n".join(lines))
        print("ERROR: no usable pairs", file=sys.stderr)
        return 1

    # ---- per-arm rates on the paired set ---------------------------------
    say("## Rates on the paired set")
    say("")
    say("| | exact match | decoded | per-number accuracy |")
    say("|---|---|---|---|")
    for arm, rows in ((bname, base), (tname, tr)):
        ex = sum(rows[i]["exact"] for i in pairs)
        de = sum(rows[i]["decoded"] for i in pairs)
        pna = statistics.mean(rows[i]["pna"] for i in pairs)
        say(f"| {arm} | {pct_ci(ex, n)} | {pct_ci(de, n)} | {pna:.3f} |")
    say("")

    # ---- primary: paired per-number accuracy ----------------------------------
    diffs = [tr[i]["pna"] - base[i]["pna"] for i in pairs]
    mean_d = statistics.mean(diffs)
    say("## Primary: paired per-number accuracy (treated − base)")
    say("")
    if n > 1 and statistics.stdev(diffs) > 0:
        sd = statistics.stdev(diffs)
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        nonzero = [d for d in diffs if d != 0]
        wil = stats.wilcoxon(nonzero).pvalue if len(nonzero) >= 1 else float("nan")
        say(f"- mean difference **{mean_d:+.3f}**, 95% CI [{mean_d - half:+.3f}, {mean_d + half:+.3f}]")
        say(f"- Wilcoxon signed-rank p = {wil:.3g} over {len(nonzero)} non-tied pairs")
    else:
        say(f"- mean difference **{mean_d:+.3f}**; every pair tied, no test possible")
    say(f"- pairs where treated > base: {sum(d > 0 for d in diffs)}, "
        f"< base: {sum(d < 0 for d in diffs)}, tied: {sum(d == 0 for d in diffs)}")
    say("")

    # ---- secondary: McNemar on exact match ------------------------------------
    df = pd.DataFrame({"t": [tr[i]["exact"] for i in pairs], "b": [base[i]["exact"] for i in pairs]})
    mc = _mcnemar_one(df["t"], df["b"])
    b_, c_ = mc["b"], mc["c"]
    p2 = stats.binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else float("nan")
    say("## Secondary: exact match, McNemar (two-sided exact)")
    say("")
    say(f"- treated-only successes b = {b_}, base-only c = {c_}"
        + (f", p = {p2:.3g}" if b_ + c_ else " — no discordant pairs, no test possible"))
    say("")

    # ---- dpn breakdown ------------------------------------------------------
    say("## By digits-per-number (per-number accuracy)")
    say("")
    say(f"| dpn | n | {bname} | {tname} | Δ |")
    say("|---|---|---|---|---|")
    for d in sorted({base[i]["dpn"] for i in pairs}):
        sub = [i for i in pairs if base[i]["dpn"] == d]
        mb = statistics.mean(base[i]["pna"] for i in sub)
        mt = statistics.mean(tr[i]["pna"] for i in sub)
        say(f"| {d or 'mixed'} | {len(sub)} | {mb:.3f} | {mt:.3f} | {mt - mb:+.3f} |")
    say("")

    # ---- monitor -----------------------------------------------------------
    say("## Monitor signals (tri-state; None = undetermined, never a failure)")
    say("")
    say("| | stego_detected | explicit_reasoning | refused |")
    say("|---|---|---|---|")
    for arm, rows in ((bname, base), (tname, tr)):
        cells = []
        for key in ("stego_detected", "explicit_reasoning", "refused"):
            v = [rows[i][key] for i in pairs]
            cells.append(f"{sum(x is True for x in v)} T / {sum(x is False for x in v)} F / "
                         f"{sum(x is None for x in v)} None")
        say(f"| {arm} | " + " | ".join(cells) + " |")
    say("")
    say(f"Logs: base `{args.base}`, treated `{args.treated}`.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\n-> {args.out}", file=sys.stderr)
    return 1 if any(leaks.values()) else 0


if __name__ == "__main__":
    sys.exit(main())

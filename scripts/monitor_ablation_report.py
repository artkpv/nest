"""
Build the RB-2 monitor-robustness table from `monitor_ablation.py` arm JSONLs.

Emits markdown tables sized for an OpenReview comment.

Denominators, and why they differ:

  payload present   the encoder actually carried the sequence
                    (`exact_match_lenient`: the digits decode correctly).
                    This is the honest denominator for a stealth claim -- on a
                    failed encoding there may be no payload to detect, so
                    counting those as "evaded the monitor" credits the encoder
                    with stealth it never exercised.
  all transcripts   stratum-weighted back to the true population mix, so
                    stratified and fully-covered arms stay comparable. The raw
                    pooled rate over a stratified arm is NOT the marginal rate.
  plain control     un-instructed answers to the same cover questions
                    (--controls-dir). Gives the false-positive rate, which the
                    counting corpus cannot supply because every transcript in it
                    came from a model told to encode. Without it a monitor
                    flagging 32% of stego is indistinguishable from one flagging
                    32% of all prose.

Per-arm tables use each arm's own coverage; paired tests (McNemar) use the
samples both arms scored.

Usage:
  python scripts/monitor_ablation_report.py --dir artifacts/2026-07-29/monitor_ablation \\
    --controls-dir artifacts/2026-07-29/monitor_controls --by-cell
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# Display order and labels. Arms absent from --dir are skipped silently.
#
# NOTE on `baseline`: this is the monitor stored in the counting `.eval` logs,
# and it is NOT a fixed Sonnet monitor. The runs set no `grader` model role, so
# `monitor_scorer` fell through `get_model(role="grader", default=get_model())`
# to the eval's own model -- each encoder graded its own transcripts. The logs
# confirm it: `stats.model_usage` names only the encoder in every counting log.
# It is therefore a *self-monitoring* arm with a different grader per cell, and
# cannot be read as a monitor-strength datapoint. `mid` is the fixed external
# Sonnet monitor that the naive gradient actually needs.
ARM_LABELS = {
    "weak": "Haiku 4.5, scheme-naive (fixed)",
    "mid": "Sonnet 4.6, scheme-naive (fixed)",
    "strong": "Opus 4.6, scheme-naive (fixed)",
    "gpt": "**GPT-5.4, scheme-naive (fixed, OpenAI)**",
    "aware": "Sonnet 4.6, scheme-AWARE (fixed)",
    "aware_strong": "Opus 4.6, scheme-AWARE (fixed)",
    "grok": "**Grok 4.3, scheme-naive (fixed, xAI)**",
    "baseline": "Self-monitor: encoder grades itself (as logged)",
}
ARM_ORDER = ["weak", "mid", "strong", "gpt", "grok", "aware", "aware_strong", "baseline"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — valid near 0/1, where the normal approx is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k/n:.1%} [{lo:.1%}, {hi:.1%}] ({k}/{n})"


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs.

    b/c are the counts where the arms disagree. Concordant pairs carry no
    information about a difference, so the test conditions on b+c and asks
    whether the split departs from 50/50. Exact (not chi-square) because the
    discordant counts here are small.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test on the 2x2, falling back to a pooled z-test.

    Fisher rather than the normal approximation because several comparisons here
    have a zero or near-zero cell (e.g. 0/250 on the negative control), which is
    exactly where the z-approximation is least trustworthy. It disagreed
    materially in practice: 100/309 vs 0/250 gives z p=3.2e-23 but Fisher
    p=4.1e-30.
    """
    if not n1 or not n2:
        return 1.0
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    except ImportError:
        pass
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def key(r: dict) -> tuple:
    return (r["log"], r["sample_id"], r["epoch"])


def load(d: Path, arms: list[str] | None) -> dict[str, dict[tuple, dict]]:
    out: dict[str, dict[tuple, dict]] = {}
    for path in sorted(d.glob("*.jsonl")):
        # `<arm>.calls.jsonl` is the raw API call log, a different schema.
        if path.name.endswith(".calls.jsonl"):
            continue
        arm = path.stem
        if arms and arm not in arms:
            continue
        recs: dict[tuple, dict] = {}
        with path.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs[key(r)] = r  # later record wins (re-grades)
        out[arm] = recs
    return out


def rate(recs: list[dict], field: str, cond=None) -> tuple[int, int]:
    """Count `field is True` over records, skipping None (undetermined).

    A monitor error or skipped grade is undetermined, never a non-detection —
    silently folding those into the denominator would understate detection.
    """
    k = n = 0
    for r in recs:
        if cond and not cond(r):
            continue
        v = r.get(field)
        if v is None:
            continue
        n += 1
        k += bool(v)
    return k, n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--by-cell", action="store_true", help="also break down per (model, D)")
    p.add_argument("--controls-dir",
                   help="directory of arm JSONLs scored on the plain negative "
                        "control (from generate_plain_controls.py)")
    p.add_argument("--reference", default="mid",
                   help="arm to use as the McNemar comparator (default: the fixed "
                        "naive Sonnet monitor, not the self-monitoring `baseline`)")
    args = p.parse_args(argv)

    data = load(Path(args.dir), args.arms)
    if not data:
        print("no arm jsonl files found")
        return 2

    present = [a for a in ARM_ORDER if a in data] + [a for a in data if a not in ARM_ORDER]

    # Exact monitor identity per arm, so a reader never has to guess which
    # "Haiku" or "Sonnet" produced a row.
    man_path = Path(args.dir) / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        print("## Monitor models\n")
        print(f"Transcripts: `{man.get('transcript_source')}` — {man.get('transcript_note')}\n")
        print(f"Provider: {man.get('provider')}. Run date: {man.get('run_date')}.\n")
        print("| Arm | Monitor model (served) | Scheme briefing | max_tokens | thinking | n | sampling |")
        print("|---|---|---|---|---|---|---|")
        for a in present:
            m = (man.get("arms") or {}).get(a, {})
            print(f"| `{a}` | `{m.get('monitor_model_served', '?')}` "
                  f"| {m.get('scheme_hint') or 'none (naive)'} "
                  f"| {m.get('max_tokens', '?')} "
                  f"| {m.get('reasoning_tokens') or 'off'} "
                  f"| {len(data[a])} | {m.get('sampling', '?')} |")
        if man.get("version_caveat"):
            print(f"\n> **Version caveat.** {man['version_caveat']}")
        bl = (man.get("arms") or {}).get("baseline", {})
        if bl.get("note"):
            print(f"\n> **`baseline` is not a fixed monitor.** {bl['note']}")
        print()

    print("## Coverage\n")
    for a in present:
        errs = sum(1 for r in data[a].values() if r.get("error"))
        print(f"- `{a}`: {len(data[a])} samples" + (f", {errs} errored" if errs else ""))

    common = set.intersection(*(set(data[a]) for a in present))
    print(f"\nPaired sample (graded by every arm): **{len(common)}**")

    enc = sum(1 for k in common if data[present[0]][k]["exact_match"])
    print(f"Of those, encoder produced a correct acrostic (`exact_match`) in "
          f"**{enc}** ({enc/len(common):.1%}).\n")

    # Population stratum sizes, taken from an arm with full coverage. Arms may
    # be stratified (all payload-carrying transcripts, a capped sample of the
    # rest), so their raw pooled rate over-weights the payload stratum and is
    # NOT the marginal detection rate. Estimate the marginal by re-weighting
    # each stratum back to its true share.
    full = max(present, key=lambda a: len(data[a]))
    pop = list(data[full].values())
    N_p = sum(1 for r in pop if r["exact_match_lenient"])
    N_a = len(pop) - N_p
    w_p, w_a = N_p / len(pop), N_a / len(pop)
    print(f"Population strata (from `{full}`, n={len(pop)}): "
          f"payload-carrying {N_p} ({w_p:.1%}), no payload {N_a} ({w_a:.1%}).\n")

    # False-positive rate from the plain negative control: same cover questions,
    # same sentence counts, same models, no encoding instruction. Without this
    # the detection rates are uninterpretable -- a monitor flagging 32% of stego
    # would be indistinguishable from one flagging 32% of all prose.
    ctl = load(Path(args.controls_dir), None) if args.controls_dir else {}
    if ctl:
        print("## False-positive rate (plain negative control)\n")
        print("_Plain answers to the same cover questions, same target sentence "
              "counts, same models, **no encoding instruction**._\n")
        print("| Monitor | flagged | FPR |")
        print("|---|---|---|")
        for a in present:
            if a not in ctl:
                continue
            k, n = rate(list(ctl[a].values()), "stego_detected")
            print(f"| {ARM_LABELS.get(a, a)} | {k}/{n} | {fmt(k, n)} |")
        print()

    print("## Detection rate by monitor\n")
    print("_Each arm uses its own full coverage. `all transcripts` is a "
          "stratum-weighted estimate, so stratified and full arms are comparable._\n")
    hdr_fpr = " FPR (plain control) |" if ctl else ""
    print("| Monitor | n graded | Detected — all transcripts (weighted) "
          f"| Detected — payload present | Stealth (payload present) |{hdr_fpr}")
    print("|---|---|---|---|---|" + ("---|" if ctl else ""))
    for a in present:
        recs = list(data[a].values())
        k_p, n_p = rate(recs, "stego_detected", cond=lambda r: r["exact_match_lenient"])
        k_a, n_a = rate(recs, "stego_detected", cond=lambda r: not r["exact_match_lenient"])
        if n_p and n_a:
            est = w_p * (k_p / n_p) + w_a * (k_a / n_a)
            se = math.sqrt(
                w_p**2 * (k_p / n_p) * (1 - k_p / n_p) / n_p
                + w_a**2 * (k_a / n_a) * (1 - k_a / n_a) / n_a
            )
            marg = f"{est:.1%} [{max(0, est - 1.96*se):.1%}, {min(1, est + 1.96*se):.1%}]"
        else:
            marg = "n/a"
        stealth = f"{1 - k_p/n_p:.1%}" if n_p else "n/a"
        fpr_cell = ""
        if ctl:
            if a in ctl:
                fk, fn = rate(list(ctl[a].values()), "stego_detected")
                fpr_cell = f" {fk/fn:.1%} ({fk}/{fn}) |" if fn else " n/a |"
            else:
                fpr_cell = " not run |"
        print(f"| {ARM_LABELS.get(a, a)} | {len(recs)} | {marg} "
              f"| {fmt(k_p, n_p)} | {stealth} |{fpr_cell}")

    print("\n## Secondary signals (paired sample, all transcripts)\n")
    print("| Monitor | explicit_reasoning | refused |")
    print("|---|---|---|")
    for a in present:
        recs = [data[a][k] for k in common]
        print(f"| {ARM_LABELS.get(a, a)} | {fmt(*rate(recs, 'explicit_reasoning'))} "
              f"| {fmt(*rate(recs, 'refused'))} |")

    if args.by_cell:
        print("\n## Detection by encoder cell (payload-carrying denominator)\n")
        cells = sorted({(data[present[0]][k]["encoder_model"], data[present[0]][k]["D"])
                        for k in common}, key=str)
        print("| Encoder | D | " + " | ".join(a for a in present) + " |")
        print("|---|---|" + "---|" * len(present))
        for model, D in cells:
            sel = [k for k in common
                   if data[present[0]][k]["encoder_model"] == model
                   and data[present[0]][k]["D"] == D]
            cellrow = []
            for a in present:
                recs = [data[a][k] for k in sel]
                k_enc, n_enc = rate(recs, "stego_detected",
                                    cond=lambda r: r["exact_match_lenient"])
                cellrow.append(f"{k_enc}/{n_enc}" if n_enc else "n/a")
            print(f"| {model} | {D} | " + " | ".join(cellrow) + " |")

    # NOT a false-positive analysis, and it must never be presented as one.
    # Every transcript in this corpus came from a model instructed to encode, so
    # there is no negative control: the right-hand column is failed/partial
    # steganography (~63% of it still decodes partially), not innocent prose.
    # The contrast therefore shows only that botched encodings are easier to
    # spot than clean ones -- expected, and NOT evidence that the monitors fail
    # to track steganography. Supporting that stronger claim would require
    # transcripts written with no encoding instruction at all.
    #
    # Split is on `exact_match_lenient` (digits decode correctly), not strict
    # `exact_match`, which additionally demands one sentence per digit -- an
    # instruction-following formality. A transcript that decodes perfectly but
    # adds a trailing sentence still carries a working payload.
    print("\n## Detection on clean vs failed encodings\n")
    print("_Both columns are steganographic attempts. The right-hand column is "
          "**failed/partial** encoding — it is **not** a negative control, since "
          "every transcript here came from a model instructed to encode. "
          + ("The false-positive rate is measured separately above, on plain "
             "un-instructed text._\n" if ctl else
             "No false-positive rate is estimable from this corpus._\n"))
    print("| Monitor | flagged, clean encoding | flagged, failed/partial | ratio | p |")
    print("|---|---|---|---|---|")
    for a in present:
        recs = list(data[a].values())  # own coverage: both strata, max n
        k_p, n_p = rate(recs, "stego_detected", cond=lambda r: r["exact_match_lenient"])
        k_n, n_n = rate(recs, "stego_detected", cond=lambda r: not r["exact_match_lenient"])
        ratio = f"{(k_p / n_p) / (k_n / n_n):.2f}x" if (n_p and n_n and k_n) else "n/a"
        p = two_prop_p(k_p, n_p, k_n, n_n)
        print(f"| {ARM_LABELS.get(a, a)} | {fmt(k_p, n_p)} | {fmt(k_n, n_n)} "
              f"| {ratio} | {p:.4f} {stars(p)} |")

    # Paired disagreement against the paper's monitor, restricted to transcripts
    # that actually carry an encoding.
    ref = args.reference if args.reference in data else (
        "mid" if "mid" in data else "baseline" if "baseline" in data else None)
    if ref:
        print(f"\n## Paired change vs `{ref}` ({ARM_LABELS.get(ref, ref)}), "
              f"payload-carrying only\n")
        print("_Same transcripts, both monitors. Exact McNemar on discordant pairs._\n")
        print(f"| Monitor | caught that {ref} missed | missed that {ref} caught "
              "| net | McNemar p |")
        print("|---|---|---|---|---|")
        base = data[ref]
        for a in present:
            if a == ref:
                continue
            gain = loss = 0
            for k in set(base) & set(data[a]):
                if not base[k]["exact_match_lenient"]:
                    continue
                b, o = base[k].get("stego_detected"), data[a][k].get("stego_detected")
                if b is None or o is None:
                    continue
                gain += (o and not b)
                loss += (b and not o)
            p = mcnemar_p(gain, loss)
            print(f"| {ARM_LABELS.get(a, a)} | {gain} | {loss} | {gain - loss:+d} "
                  f"| {p:.4f} {stars(p)} |")

    if man_path.exists():
        lim = json.loads(man_path.read_text()).get("known_limitations") or {}
        if lim:
            print("\n## Known limitations\n")
            for k, v in lim.items():
                print(f"- **{k}** — {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

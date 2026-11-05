"""
Reproducibility ledger: model identifiers, inference settings, API calls, tokens, cost.

Answers the "what exactly did you run" question over the whole project. Three
harnesses produced model calls and they account for themselves differently, so
all three are walked separately and then summed:

  A. Inspect `.eval` logs (`logs/`, `artifacts/**`). Exact token counts and one
     `ModelEvent` per API call. **Deduplicated by file basename** — re-scoring
     passes (`rescore_counting_logs.py`, `regrade_monitor.py`) copy whole logs
     to a new directory, so `artifacts/2026-07-30/word_acrostic_regrade/` and
     `artifacts/2026-05-01/sweep2/` contain the same runs. Counting both double
     counts ~70 files.
  B. The standalone counting runner removed in `29f671ad`
     (`artifacts/stego_counting/`). Trial JSONs store prompt and response text
     but **no token counts**, so tokens are estimated from character length.
  C. The RB-2 monitor work (`artifacts/2026-07-29/`). `regrade_monitor.py` and
     `monitor_ablation.py` call the monitor through `get_model()` outside the
     Inspect eval loop, so those calls leave **no ModelEvent** and are invisible
     to A. Arms with a `.calls.jsonl` carry real usage; the rest are estimated.
     The `baseline` arm is read out of the logs and makes no calls.

Cost is an estimate. `total_cost` is None on every log we hold — no provider
returned a price — so PRICES below is an input assumption, not a measurement.
Token counts are exact for A; everything else is marked estimated in the output.

Usage:
  python scripts/repro_ledger.py                     # summary tables
  python scripts/repro_ledger.py --json out.json     # machine-readable
  python scripts/repro_ledger.py --per-model         # full per-model breakdown
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import sys

from inspect_ai.log import read_eval_log

# USD per 1M tokens (input, output), by served model id. Tier list prices.
# VERIFY THESE before quoting a cost anywhere — they are the single largest
# source of error in the total, far larger than the token estimates.
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15, 75), "claude-opus-4-6": (15, 75),
    "claude-opus-4-5-20251101": (15, 75), "claude-opus-4-1-20250805": (15, 75),
    "claude-opus-4-20250514": (15, 75),
    "claude-sonnet-4-6": (3, 15), "claude-sonnet-4-5": (3, 15),
    "claude-sonnet-4-5-20250929": (3, 15), "claude-sonnet-4-20250514": (3, 15),
    "claude-3-7-sonnet-20250219": (3, 15), "claude-3-5-sonnet-20241022": (3, 15),
    "claude-3-sonnet-20240229": (3, 15), "sonnet-3-7": (3, 15),
    "claude-haiku-4-5-20251001": (1, 5), "claude-haiku-4-5": (1, 5),
    "claude-3-5-haiku-20241022": (0.8, 4), "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-5.4": (1.25, 10), "gpt-5.4-mini": (0.25, 2),
    "gpt-5.2": (1.25, 10), "gpt-5.2-2025-12-11": (1.25, 10),
    "gpt-5.1-2025-11-13": (1.25, 10), "gpt-5-2025-08-07": (1.25, 10),
    "gpt-5-mini-2025-08-07": (0.25, 2), "gpt-5-nano-2025-08-07": (0.05, 0.4),
    "gpt-4.1-2025-04-14": (2, 8), "gpt-4.1-mini-2025-04-14": (0.4, 1.6),
    "gpt-4.1-nano-2025-04-14": (0.1, 0.4),
    "gpt-4o-2024-05-13": (5, 15), "gpt-4o-2024-08-06": (2.5, 10),
    "gpt-4o-2024-11-20": (2.5, 10), "gpt-4o": (2.5, 10),
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4-0613": (30, 60), "gpt-4": (30, 60),
    "gpt-4-turbo-2024-04-09": (10, 30), "gpt-4-turbo": (10, 30),
    "gpt-3.5-turbo-0125": (0.5, 1.5), "gpt-3.5-turbo": (0.5, 1.5),
    # Google
    "gemini-2.5-flash": (0.3, 2.5), "gemini-2.5-pro": (1.25, 10),
    "gemini-2.0-flash": (0.1, 0.4), "gemini-1.5-flash": (0.075, 0.3),
    "gemini-1.5-pro": (1.25, 5), "gemini-3-pro-preview": (2, 12),
    # OpenRouter-hosted
    "deepseek-v3.2": (0.28, 0.42), "deepseek-chat-v3-0324": (0.28, 0.88),
    "qwen3-235b-a22b-2507": (0.2, 0.6), "kimi-k2": (0.6, 2.5),
    "grok-4.1-fast": (0.2, 0.5), "grok-code-fast-1": (0.2, 1.5),
    "grok-4.20-0309-non-reasoning": (3, 15), "grok-4.3": (3, 15),
}

CHARS_PER_TOKEN = 4.0  # only for sources with no usage accounting (B, part of C)

# Measured from the Inspect Sonnet-4.6 grader; used to price the legacy monitor
# passes, which recorded a verdict but not the call.
_GRADER_AVG = (674, 302)


def _cost(model: str, tin: float, tout: float) -> float | None:
    # Some rows carry a trailing " (annotation)" to distinguish passes over the
    # same model; strip it before the price lookup.
    key = model.split(" (")[0].split("/")[-1]
    p = PRICES.get(key)
    return None if p is None else tin / 1e6 * p[0] + tout / 1e6 * p[1]


def _eval_paths() -> list[str]:
    return sorted(glob.glob("logs/*.eval")) + sorted(glob.glob("artifacts/**/*.eval", recursive=True))


def source_a() -> dict:
    """Inspect logs. Exact tokens; deduplicated by basename."""
    seen: set[str] = set()
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    resolved: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    settings: collections.Counter = collections.Counter()
    dirty: collections.Counter = collections.Counter()
    packages: collections.Counter = collections.Counter()
    dates: list[str] = []
    wall = dt.timedelta()
    dups = samples = 0

    for p in _eval_paths():
        base = os.path.basename(p)
        if base in seen:
            dups += 1
            continue
        seen.add(base)
        log = read_eval_log(p)
        dates.append(log.eval.created[:10])
        samples += (log.results.total_samples if log.results else 0) or 0
        rev = log.eval.revision
        dirty[str(rev.dirty) if rev else "no-git"] += 1
        packages[str((log.eval.packages or {}).get("inspect_ai"))] += 1
        try:
            wall += (dt.datetime.fromisoformat(log.stats.completed_at)
                     - dt.datetime.fromisoformat(log.stats.started_at))
        except Exception:  # noqa: BLE001 — a cancelled run may lack a timestamp
            pass
        for s in (log.samples or []):
            for e in (s.events or []):
                if e.event != "model":
                    continue
                role = str(getattr(e, "role", None) or "encoder")
                c = agg[e.model]
                c["calls"] += 1
                c[f"{role}_calls"] += 1
                resolved[e.model][str(getattr(e.output, "model", None))] += 1
                u = getattr(e.output, "usage", None)
                if u is not None:
                    c["in"] += u.input_tokens or 0
                    c["out"] += u.output_tokens or 0
                    c["reasoning"] += u.reasoning_tokens or 0
                gc = getattr(e, "config", None)
                settings[(e.model.split("/")[0], role,
                          getattr(gc, "temperature", None), getattr(gc, "max_tokens", None),
                          getattr(gc, "top_p", None), getattr(gc, "reasoning_effort", None),
                          getattr(gc, "reasoning_tokens", None))] += 1

    return {"models": agg, "resolved": resolved, "settings": settings, "dirty": dirty,
            "packages": packages, "logs": len(seen), "dup_files": dups, "samples": samples,
            "first": min(dates), "last": max(dates), "wall": wall, "estimated": False}


def source_b() -> dict:
    """Legacy standalone counting runner. Tokens estimated from characters."""
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    dates: list[str] = []
    for p in glob.glob("artifacts/stego_counting/*/trials/*.json"):
        if p.endswith(".monitor.json"):
            continue
        try:
            t = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        c = agg[t.get("model") or "?"]
        c["calls"] += 1
        c["in"] += len(t.get("prompt") or "") / CHARS_PER_TOKEN
        c["out"] += len(t.get("response") or "") / CHARS_PER_TOKEN
        if t.get("timestamp"):
            dates.append(t["timestamp"][:10])

    n_mon = len(glob.glob("artifacts/stego_counting/*/trials/*.monitor.json"))
    mon = agg["anthropic/claude-sonnet-4-6 (retro-monitor)"]
    mon["calls"] += n_mon
    mon["in"] += n_mon * _GRADER_AVG[0]
    mon["out"] += n_mon * _GRADER_AVG[1]

    return {"models": agg, "monitor_calls": n_mon,
            "first": min(dates) if dates else None, "last": max(dates) if dates else None,
            "estimated": True}


def source_c() -> dict:
    """RB-2 monitor ablation + controls: calls made outside the Inspect eval loop."""
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for d in ("artifacts/2026-07-29/monitor_ablation", "artifacts/2026-07-29/monitor_controls"):
        man_path = f"{d}/manifest.json"
        man = json.load(open(man_path)) if os.path.exists(man_path) else {}
        for res in sorted(glob.glob(f"{d}/*.jsonl")):
            if res.endswith(".calls.jsonl"):
                continue
            arm = os.path.basename(res)[:-6]
            if arm == "baseline":
                continue  # read straight out of the logs; no API call
            calls_path = res[:-6] + ".calls.jsonl"
            if os.path.exists(calls_path):
                for line in open(calls_path):
                    r = json.loads(line)
                    u = r.get("usage") or {}
                    c = agg[r.get("monitor_model") or "?"]
                    c["calls"] += 1
                    c["in"] += u.get("input_tokens") or 0
                    c["out"] += u.get("output_tokens") or 0
            else:
                model = ((man.get("arms") or {}).get(arm) or {}).get(
                    "monitor_model_requested", f"?unknown ({arm})")
                agg[model]["calls_est"] += sum(1 for _ in open(res))

    ctl = "data/plain_controls_2026-07-29.jsonl"
    n_ctl = 0
    if os.path.exists(ctl):
        for line in open(ctl):
            r = json.loads(line)
            c = agg[r["model"]]
            c["calls"] += 1
            c["in"] += len(r.get("prompt") or "") / CHARS_PER_TOKEN
            c["out"] += len(r.get("response") or "") / CHARS_PER_TOKEN
            n_ctl += 1

    # Arms without a call log: charge them the same model's measured average.
    n_est = 0
    for c in agg.values():
        if c["calls_est"]:
            n_est += c["calls_est"]
            if c["calls"]:
                c["in"] += c["calls_est"] * c["in"] / c["calls"]
                c["out"] += c["calls_est"] * c["out"] / c["calls"]
            c["calls"] += c["calls_est"]

    return {"models": agg, "estimated_calls": n_est, "control_generations": n_ctl,
            "estimated": True}


def _table(agg, per_model: bool, label: str) -> tuple[int, float, float, float]:
    calls = tin = tout = usd = 0.0
    unpriced = []
    for m, c in sorted(agg.items(), key=lambda kv: -(kv[1]["in"] + kv[1]["out"])):
        cost = _cost(m, c["in"], c["out"])
        if cost is None:
            unpriced.append(m)
        if per_model:
            print(f"  {m:44s} {c['calls']:7,.0f} {c['in']:12,.0f} {c['out']:11,.0f} "
                  f"{('$' + format(cost, ',.2f')) if cost is not None else 'UNPRICED':>10s}")
        calls += c["calls"]; tin += c["in"]; tout += c["out"]; usd += cost or 0
    if unpriced:
        print(f"  !! no price for: {', '.join(unpriced)}", file=sys.stderr)
    print(f"  {label:44s} {calls:7,.0f} {tin:12,.0f} {tout:11,.0f} {'$' + format(usd, ',.2f'):>10s}")
    return int(calls), tin, tout, usd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-model", action="store_true", help="full per-model breakdown")
    p.add_argument("--json", help="also write the ledger as JSON")
    args = p.parse_args(argv)

    a, b, c = source_a(), source_b(), source_c()

    print(f"{'':46s} {'calls':>7s} {'input tok':>12s} {'output tok':>11s} {'est. USD':>10s}")
    print("\nA. Inspect .eval logs  (tokens EXACT)")
    ta = _table(a["models"], args.per_model, "subtotal")
    print("\nB. Legacy counting runner  (tokens ESTIMATED from chars)")
    tb = _table(b["models"], args.per_model, "subtotal")
    print("\nC. RB-2 monitor calls outside the eval loop  (partly ESTIMATED)")
    tc = _table(c["models"], args.per_model, "subtotal")

    print(f"\n{'TOTAL':46s} {ta[0]+tb[0]+tc[0]:7,d} {ta[1]+tb[1]+tc[1]:12,.0f} "
          f"{ta[2]+tb[2]+tc[2]:11,.0f} {'$' + format(ta[3]+tb[3]+tc[3], ',.2f'):>10s}")

    print(f"\nA: {a['logs']} logs ({a['dup_files']} duplicate copies skipped), "
          f"{a['samples']:,} samples, {a['first']}..{a['last']}, "
          f"wall-clock {a['wall']}")
    print(f"   git worktree dirty at run time: {dict(a['dirty'])}")
    print(f"   inspect_ai versions: {dict(a['packages'])}")
    print(f"B: {b['first']}..{b['last']}, {b['monitor_calls']:,} retro-monitor calls")
    print(f"C: {c['estimated_calls']:,} monitor calls with no usage log (estimated), "
          f"{c['control_generations']} control generations")
    print(f"\nCost uses the PRICES table in this file — an assumption, not a measurement. "
          f"No provider returned a price on any log.")

    print("\n== requested -> served model id ==")
    for req in sorted(a["resolved"]):
        for served, n in a["resolved"][req].most_common():
            flag = "  (no dated snapshot)" if served.split("/")[-1] == req.split("/")[-1] else ""
            print(f"  {req:42s} -> {served:32s} {n:7,d}{flag}")

    print("\n== inference settings actually sent (provider x role) ==")
    print(f"  {'provider':10s} {'role':8s} {'temp':6s} {'max_tok':8s} {'top_p':6s} "
          f"{'r_effort':9s} {'r_tokens':9s} {'calls':>7s}")
    for k, v in sorted(a["settings"].items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])):
        print(f"  {k[0]:10s} {k[1]:8s} {str(k[2]):6s} {str(k[3]):8s} {str(k[4]):6s} "
              f"{str(k[5]):9s} {str(k[6]):9s} {v:7,d}")

    if args.json:
        json.dump({
            "totals": {"calls": ta[0] + tb[0] + tc[0], "input": ta[1] + tb[1] + tc[1],
                       "output": ta[2] + tb[2] + tc[2], "usd": ta[3] + tb[3] + tc[3]},
            "sources": {
                "A": {"calls": ta[0], "in": ta[1], "out": ta[2], "usd": ta[3],
                      "logs": a["logs"], "samples": a["samples"], "dup_files": a["dup_files"],
                      "first": a["first"], "last": a["last"], "wall": str(a["wall"]),
                      "dirty": dict(a["dirty"]), "packages": dict(a["packages"])},
                "B": {"calls": tb[0], "in": tb[1], "out": tb[2], "usd": tb[3],
                      "first": b["first"], "last": b["last"]},
                "C": {"calls": tc[0], "in": tc[1], "out": tc[2], "usd": tc[3],
                      "estimated_calls": c["estimated_calls"]},
            },
            "resolved": {k: dict(v) for k, v in a["resolved"].items()},
            "prices": PRICES,
        }, open(args.json, "w"), indent=1, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

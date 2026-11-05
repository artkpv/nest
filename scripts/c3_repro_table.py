"""
Per-experiment reproducibility table (ROADMAP C3). Offline: reads logs only.

Reuses the rebuttal ledger (`repro_ledger.py`) unchanged: the same .eval walk with
basename dedup (source A), the legacy counting runner (B) and the RB-2 monitor work
(C). Adds what that ledger did not split out: one row per experiment the paper
reports, and the calls made outside .eval logs (calibration, preflight, re-grade
and probe call files).

Two steps, so the slow one runs once:
  python scripts/c3_repro_table.py scan  --out artifacts/2026-11-05/c3_repro   # per-log facts -> logs.jsonl
  python scripts/c3_repro_table.py table --out artifacts/2026-11-05/c3_repro   # rows -> experiments.csv

Cost: tokens x a price table. No provider returned a price, so cost is an estimate
under two tables: the rebuttal's (`repro_ledger.PRICES`, Aug 2026) and the clean
re-run's (`stego_inspect.spend.PRICES`, Oct 2026). For the camera-ready runs the
spend ledger (`spend.run_spend`) is quoted as recorded.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inspect_ai.log import read_eval_log

sys.path.insert(0, os.path.dirname(__file__))
import repro_ledger  # noqa: E402
from stego_inspect import spend  # noqa: E402

REBUTTAL_CUTOFF = "2026-07-30"  # last run date in the rebuttal figure (REPRODUCIBILITY.md §3)

# First match wins. (row, path prefix, extra test on the log facts or None.)
# Rows are the experiments the paper reports; everything else falls into OTHER.
OTHER = "Development, pilots and rebuttal-only runs"
OTHER_LATE = "Probes after the rebuttal, not reported"
RULES: list[tuple[str, str, object]] = [
    ("Channel screening", "artifacts/2026-04-23/main_sweep/screen_logs", None),
    ("Channel screening", "artifacts/2026-05-01/sweep2/screen_logs", None),
    ("Channel screening", "artifacts/2026-05-04/v3_mapping_off/screen_logs", None),
    ("Channel screening", "artifacts/2026-04-23/bitchannel_evidence", None),
    ("Counting sweep", "logs/", lambda f: f["hidden_dataset"] == "counting"),
    ("Counting sweep", "artifacts/2026-07-28/counting_topup", None),
    ("RG sweep, April-May (submission)", "artifacts/2026-04-23/main_sweep", None),
    ("RG sweep, April-May (submission)", "artifacts/2026-04-23/opus46_sweep_mapping", None),
    ("RG sweep, April-May (submission)", "artifacts/2026-05-04/", None),
    ("Truncation probe", "artifacts/2026-09-24/truncation_probe", None),
    ("B4 open-weight counting", "artifacts/2026-09-16/openweight_probe", None),
    ("B4 open-weight counting", "artifacts/2026-10-21/openweight_qwen38_reasoning_off", None),
    ("B4 open-weight counting", "artifacts/2026-11-04/b4_openweight_gemma_llama", None),
    ("B6 clean RG re-run", "artifacts/2026-10-31/", None),
    ("B5 first-word arm", "artifacts/2026-11-04/rg_clean_first_word", None),
    ("B8 Opus 4.6 maze follow-up", "artifacts/2026-11-04/rg_opus46_maze_followup", None),
]

# Calls outside .eval logs: (row, glob, kind). kind picks the reader below.
EXTRA = [
    ("RG sweep, April-May (submission)", "artifacts/2026-04-23/main_sweep/calibration/calls/*/*.jsonl", "calibration"),
    ("RG sweep, April-May (submission)", "artifacts/2026-05-04/sweep3/calibration/calls/*/*.jsonl", "calibration"),
    (OTHER, "artifacts/2026-04-2[12]/*/calibration/calls/*/*.jsonl", "calibration"),
    (OTHER, "artifacts/2026-04-23/rg_experiment/calibration/calls/*/*.jsonl", "calibration"),
    (OTHER, "artifacts/2026-04-23/opus46_sweep_mapping/calibration/calls/*/*.jsonl", "calibration"),
    (OTHER, "artifacts/2026-05-01/sweep2/calibration/calls/*/*.jsonl", "calibration"),
    ("B1 ttc re-run", "artifacts/2026-10-30/rg_ttc_rerun/calls.jsonl", "chars"),
    ("Truncation probe", "artifacts/2026-09-24/truncation_probe/calls.jsonl", "probe"),
    ("B6 clean RG re-run", "artifacts/2026-10-31/rg_clean/calibration/calls/*/*.jsonl", "calibration"),
    ("B6 clean RG re-run", "artifacts/2026-10-31/rg_clean/preflight/calls.jsonl", "usage"),
    ("B6 clean RG re-run", "artifacts/2026-10-31/rg_clean*/eval_logs_sensitivity/usage.jsonl", "usage"),
    ("B5 first-word arm", "artifacts/2026-11-04/rg_clean_first_word/eval_logs_sensitivity/usage.jsonl", "usage"),
    ("B8 Opus 4.6 maze follow-up", "artifacts/2026-11-04/rg_opus46_maze_followup/calibration/calls/*/*.jsonl", "calibration"),
    ("B8 Opus 4.6 maze follow-up", "artifacts/2026-11-04/rg_opus46_maze_followup/preflight/calls.jsonl", "usage"),
    ("B4 open-weight counting", "artifacts/2026-11-04/b4_openweight_gemma_llama/monitor/*.calls.jsonl", "usage"),
]

# Spend ledgers as recorded for the camera-ready runs (rg_spend.py).
SPEND_DIRS = {
    "B6 clean RG re-run": ["artifacts/2026-10-31/rg_clean", "artifacts/2026-10-31/rg_clean_mapping",
                           "artifacts/2026-10-31/opus46_cut_sweep"],
    "B5 first-word arm": ["artifacts/2026-11-04/rg_clean_first_word"],
    "B8 Opus 4.6 maze follow-up": ["artifacts/2026-11-04/rg_opus46_maze_followup"],
}


def _log_facts(path: str) -> dict:
    log = read_eval_log(path)
    args = log.eval.task_args or {}
    f = {
        "path": path, "task": log.eval.task, "created": log.eval.created[:10],
        "model": log.eval.model, "base_url": log.eval.model_base_url,
        "hidden_dataset": args.get("hidden_dataset"), "algorithm": args.get("algorithm"),
        "status": log.status, "samples": len(log.samples or []),
        "inspect_ai": (log.eval.packages or {}).get("inspect_ai"),
        "calls": 0, "in": 0, "out": 0, "reasoning": 0,
        "served": collections.Counter(), "settings": collections.Counter(), "by_model": {},
    }
    by_model: dict = collections.defaultdict(lambda: [0, 0, 0])
    for s in log.samples or []:
        for e in s.events or []:
            if e.event != "model":
                continue
            u = getattr(e.output, "usage", None)
            tin, tout = (u.input_tokens or 0, u.output_tokens or 0) if u else (0, 0)
            f["calls"] += 1
            f["in"] += tin
            f["out"] += tout
            f["reasoning"] += (u.reasoning_tokens or 0) if u else 0
            by_model[e.model][0] += 1
            by_model[e.model][1] += tin
            by_model[e.model][2] += tout
            f["served"][f"{e.model} -> {getattr(e.output, 'model', None)}"] += 1
            c = e.config
            role = "encoder" if e.model == log.eval.model else "grader"
            f["settings"][json.dumps([role, e.model, c.temperature, c.max_tokens,
                                      c.reasoning_effort, c.reasoning_tokens, c.seed])] += 1
    f["by_model"] = dict(by_model)
    f["served"], f["settings"] = dict(f["served"]), dict(f["settings"])
    return f


def scan(out: Path) -> None:
    paths, seen = [], set()
    for p in repro_ledger._eval_paths():  # same order and dedup as the rebuttal ledger
        base = os.path.basename(p)
        if base not in seen:
            seen.add(base)
            paths.append(p)
    with ProcessPoolExecutor() as ex, open(out / "logs.jsonl", "w") as fh:
        for f in ex.map(_log_facts, paths, chunksize=4):
            fh.write(json.dumps(f) + "\n")
    print(f"{len(paths)} logs -> {out / 'logs.jsonl'}")


def row_of(f: dict) -> str:
    for row, prefix, test in RULES:
        if f["path"].startswith(prefix) and (test is None or test(f)):
            return row
    return OTHER if f["created"] <= REBUTTAL_CUTOFF else OTHER_LATE


def _extra_calls(kind: str, path: str) -> tuple[int, int, int, dict, set]:
    """(calls, in, out, by_model, dates) from a call file. Tokens 0 where none were logged."""
    calls = tin = tout = 0
    dates: set = set()
    by_model: dict = collections.defaultdict(lambda: [0, 0, 0])
    for line in open(path):
        r = json.loads(line)
        if r.get("ts"):
            dates.add(r["ts"][:10])
        if kind == "calibration":
            arms = [a for a in ("direct", "cot", "ttc") if isinstance(r.get(a), dict)]
            for a in arms:
                i, o = r[a].get("input_tokens") or 0, r[a].get("output_tokens") or 0
                m = r.get("model") or "not recorded"
                calls += 1; tin += i; tout += o
                by_model[m][0] += 1; by_model[m][1] += i; by_model[m][2] += o
        elif kind in ("chars", "probe"):
            # chars: no usage logged, tokens from characters as for the legacy runner (estimate).
            if kind == "chars":
                i = sum(len(str(m.get("content") or "")) for m in r.get("messages") or []) / repro_ledger.CHARS_PER_TOKEN
                o = len(str(r.get("completion") or "")) / repro_ledger.CHARS_PER_TOKEN
            else:
                i, o = r.get("in_tok") or 0, r.get("out_tok") or 0
            m = r.get("model") or "not recorded"
            calls += 1; tin += i; tout += o
            by_model[m][0] += 1; by_model[m][1] += i; by_model[m][2] += o
        else:
            u = r.get("usage") if isinstance(r.get("usage"), dict) else r
            i, o = u.get("input_tokens") or 0, u.get("output_tokens") or 0
            m = r.get("model") or r.get("monitor_model") or "not recorded"
            calls += 1; tin += i; tout += o
            by_model[m][0] += 1; by_model[m][1] += i; by_model[m][2] += o
    return calls, tin, tout, dict(by_model), dates


def _price(table: dict, model: str, tin: float, tout: float) -> float | None:
    name = model.split(" (")[0].rsplit("/", 1)[-1]
    hits = [k for k in table if name.startswith(k)]
    if not hits:
        return None
    p = table[max(hits, key=len)]
    return tin / 1e6 * p[0] + tout / 1e6 * p[1]


def table(out: Path) -> None:
    logs = [json.loads(line) for line in open(out / "logs.jsonl")]
    rows: dict = collections.defaultdict(lambda: {
        "logs": 0, "samples": 0, "calls": 0, "in": 0, "out": 0, "extra_calls": 0, "extra_untokened": 0,
        "dates": set(), "served": collections.Counter(), "settings": collections.Counter(),
        "by_model": collections.defaultdict(lambda: [0, 0, 0]), "inspect_ai": collections.Counter(),
        "in_rebuttal_calls": 0})
    per_log = []
    for f in logs:
        r = rows[row_of(f)]
        r["logs"] += 1; r["samples"] += f["samples"]; r["calls"] += f["calls"]
        r["in"] += f["in"]; r["out"] += f["out"]; r["dates"].add(f["created"])
        r["served"].update(f["served"]); r["settings"].update(f["settings"])
        r["inspect_ai"][f["inspect_ai"]] += 1
        if f["created"] <= REBUTTAL_CUTOFF:
            r["in_rebuttal_calls"] += f["calls"]
        for m, v in f["by_model"].items():
            for i in range(3):
                r["by_model"][m][i] += v[i]
        per_log.append({"row": row_of(f), **{k: f[k] for k in ("path", "created", "model", "task", "hidden_dataset",
                                                               "samples", "calls", "in", "out")}})

    for row, pattern, kind in EXTRA:
        for p in sorted(glob.glob(pattern)):
            c, i, o, bm, ds = _extra_calls(kind, p)
            r = rows[row]
            r["dates"].update(ds)
            r["extra_calls"] += c; r["calls"] += c; r["in"] += i; r["out"] += o
            if i + o == 0:
                r["extra_untokened"] += c
            if kind == "chars":
                r["estimated_tokens"] = True
            for m, v in bm.items():
                for k in range(3):
                    r["by_model"][m][k] += v[k]

    # Legacy runner (B) and RB-2 monitor work (C), as the rebuttal ledger counts them.
    b, c = repro_ledger.source_b(), repro_ledger.source_c()
    for row, src, dates in (("Counting sweep", b, (b["first"], b["last"])),
                            ("Counting monitor re-grade (RB-2)", c, ("2026-07-29", "2026-07-29"))):
        r = rows[row]
        for m, v in src["models"].items():
            r["calls"] += v["calls"]; r["in"] += v["in"]; r["out"] += v["out"]
            r["in_rebuttal_calls"] += v["calls"]
            r["by_model"][m][0] += v["calls"]; r["by_model"][m][1] += v["in"]; r["by_model"][m][2] += v["out"]
        r["dates"].update(d for d in dates if d)
        r["estimated_tokens"] = True

    out_rows = []
    for name, r in rows.items():
        usd_aug = usd_oct = 0.0
        unpriced = set()
        for m, (n, i, o) in r["by_model"].items():
            a, b_ = _price(repro_ledger.PRICES, m, i, o), _price(spend.PRICES, m, i, o)
            usd_aug += a or 0
            usd_oct += b_ if b_ is not None else (a or 0)  # older models: only the August table prices them
            if a is None and b_ is None and (i or o):
                unpriced.add(m)
        recorded = None
        if name in SPEND_DIRS:
            recorded = sum(spend.run_spend(Path(d))["usd"] for d in SPEND_DIRS[name])
        d = sorted(r["dates"])
        out_rows.append({
            "experiment": name, "first": d[0] if d else "", "last": d[-1] if d else "",
            "logs": r["logs"], "samples": r["samples"], "calls": int(r["calls"]),
            "calls_outside_logs": r["extra_calls"], "calls_without_tokens": r["extra_untokened"],
            "calls_in_rebuttal_figure": int(r["in_rebuttal_calls"]),
            "in": int(r["in"]), "out": int(r["out"]),
            "tokens_estimated": bool(r.get("estimated_tokens")),
            "usd_aug_prices": round(usd_aug, 2), "usd_oct_prices": round(usd_oct, 2),
            "usd_spend_ledger": None if recorded is None else round(recorded, 2),
            "unpriced": ";".join(sorted(unpriced)),
            "inspect_ai": ";".join(f"{k}:{v}" for k, v in r["inspect_ai"].items()),
            "models": json.dumps({m: v for m, v in sorted(r["by_model"].items(), key=lambda kv: -kv[1][0])}),
            "served": json.dumps(dict(r["served"].most_common())),
            "settings": json.dumps(dict(r["settings"].most_common())),
        })
    import pandas as pd

    df = pd.DataFrame(out_rows).sort_values("first")
    df.to_csv(out / "experiments.csv", index=False)
    pd.DataFrame(per_log).to_csv(out / "logs_by_row.csv", index=False)
    print(df[["experiment", "first", "last", "logs", "samples", "calls", "calls_in_rebuttal_figure", "in", "out",
              "usd_aug_prices", "usd_oct_prices", "usd_spend_ledger"]].to_string(index=False))
    print(f"\nwrote {out / 'experiments.csv'}, {out / 'logs_by_row.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("step", choices=["scan", "table"])
    ap.add_argument("--out", type=Path, default=Path("artifacts/2026-11-05/c3_repro"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scan(a.out) if a.step == "scan" else table(a.out)


if __name__ == "__main__":
    main()

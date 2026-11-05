#!/usr/bin/env python3
"""
Explanation sweep for the content_filter cuts on Opus 4.6. PAID: about 150 calls, $2.

Re-sends the stego prompt on the exact 150 published Opus 4.6 items, stego arm only,
with no baselines, monitor or translator. The published run on these items had no
cuts, so this measures the change in provider behaviour on identical prompts, and
it records `stop_details.category`, which only an eval log with --log-model-api
keeps. Not a result cell: no stego_success comes out of it.

Usage (from code/):
    python scripts/rg_cut_sweep.py run --out-dir artifacts/<date>/opus46_cut_sweep   # safe to re-run
    python scripts/rg_cut_sweep.py analyze --out-dir artifacts/<date>/opus46_cut_sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

ITEMS = Path("artifacts/2026-10-30/rg_ttc_rerun/items.jsonl")
MODEL = "anthropic/claude-opus-4-6"


def write_pools(out: Path) -> dict[str, Path]:
    """One pool per family, in published order, so cover questions line up too."""
    items = defaultdict(list)
    for line in open(ITEMS):
        r = json.loads(line)
        items[r["family"]].append(r)
    pools = {}
    for family, rows in items.items():
        rows.sort(key=lambda r: int(r["sample_id"].split("_")[1]))
        pool = out / "pools" / f"rg_{family}_opus46cut_edge.jsonl"
        pool.parent.mkdir(parents=True, exist_ok=True)
        with open(pool, "w") as f:
            for r in rows:
                f.write(json.dumps({
                    "question": r["problem"], "answer": r["target"],
                    "metadata": {"source_dataset": family},
                }) + "\n")
        pool.with_suffix(".json").write_text(json.dumps(list(range(len(rows)))))
        pools[family] = pool
    return pools


def family_done(out: Path, family: str) -> bool:
    """True if a finished log exists. Unfinished ones move away, so re-runs cannot
    repeat paid calls or double-count samples."""
    done = False
    for info in list_eval_logs(str(out / "eval_logs" / family)):
        src = Path(info.name.removeprefix("file://"))
        try:
            finished = read_eval_log(info.name, header_only=True).status == "success"
        except Exception:  # noqa: BLE001  # a killed process leaves an unreadable file
            finished = False
        if finished:
            done = True
        else:
            dest = out / "eval_logs_failed" / family
            dest.mkdir(parents=True, exist_ok=True)
            src.rename(dest / src.name)
    return done


def run(out: Path) -> int:
    # `inspect eval` loads the task from its own directory, so paths must be absolute.
    out = out.resolve()
    failed = 0
    for family, pool in write_pools(out).items():
        if family_done(out, family):
            print(f"SKIP  {family}: finished log exists")
            continue
        rc = subprocess.run([
            "inspect", "eval", "stego_inspect/tasks.py@stego_cot_eval",
            "--model", MODEL,
            "-T", "hidden_dataset=reasoning_gym",
            "-T", f"dataset_path={pool}",
            "-T", f"sample_ids_file={pool.with_suffix('.json')}",
            "-T", "algorithm=sentence_acrostic",
            "-T", "use_mapping=false",
            "-T", "include_monitor=false",
            "-T", "include_translation=false",
            "-T", "clean_protocol=true",
            "-T", "n_samples=50",
            "--log-model-api",
            "--fail-on-error", "0.1",
            "--retry-on-error=2",
            "--log-dir", str(out / "eval_logs" / family),
            "--max-connections", "10",
        ]).returncode
        failed += rc != 0
    return analyze(out) if not failed else 1


def stop_category(sample) -> str | None:
    """stop_details.category of the stego call, from the raw response in the log."""
    events = [e for e in sample.events if e.event == "model"]
    if not events or events[0].call is None:
        return None
    return ((events[0].call.response or {}).get("stop_details") or {}).get("category")


def cut_categories(log_dir: Path) -> list[dict]:
    """One row per (model, family, stop_reason, category, HIDDEN_ANSWER present)."""
    counts: Counter = Counter()
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        for s in log.samples or []:
            stop = str(s.output.stop_reason) if s.output and s.output.choices else None
            cut = stop in ("content_filter", "max_tokens", "model_length")
            counts[(
                log.eval.model,
                (s.metadata or {}).get("hidden_task_type"),
                stop,
                stop_category(s) if cut else None,
            )] += 1
    return [
        {"model": m, "family": f, "stop_reason": st, "category": c, "n": n}
        for (m, f, st, c), n in sorted(counts.items(), key=str)
    ]


def analyze(out: Path) -> int:
    rows = cut_categories(out / "eval_logs")
    path = out / "cut_categories.json"
    path.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)
    total = sum(r["n"] for r in rows)
    cut = sum(r["n"] for r in rows if r["stop_reason"] != "stop")
    print(f"cut {cut}/{total}; published run on these items had none. Written to {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("command", choices=["run", "analyze"])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    return run(a.out_dir) if a.command == "run" else analyze(a.out_dir)


if __name__ == "__main__":
    sys.exit(main())

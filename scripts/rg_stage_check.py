#!/usr/bin/env python3
"""
Stage check for the clean re-run: is the run healthy enough to go on? Offline.

Shows coverage only, never stego_correct or stego_success, so nobody reads results
or tunes anything mid-run. Exit code 1 means a stop rule fired.

Usage (from code/):
    python scripts/rg_stage_check.py calibration artifacts/<date>/rg_clean
    python scripts/rg_stage_check.py eval artifacts/<date>/rg_clean
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from rg_clean_report import baseline_status, col, load_samples

MIN_POOL = 50                 # items needed for n=50
STOP_POOL = 30                # below this a cell cannot carry the comparison
MAX_UNDETERMINED_ARM = 0.2    # direct or filler in the final pool
WARN_CUT = 0.1
STOP_CUT = 0.3                # assessment §8: RG would join counting's bias problem
STOP_MONITOR_FAILED = 0.1
STOP_BASELINE_ERRORS = 0.1    # an outage mid-cell, which the eval itself survives
# Set per model by `clean_protocol.stop_cut` in the config, for a known cut rate.
EXPECTED_SMALL = {"gpt54miniclean__maze"}   # CoT 29% in the published sweep


def check_calibration(run: Path) -> int:
    rows, stops = [], []
    for path in sorted((run / "calibration").glob("*__*.json")):
        r = json.loads(path.read_text())
        final, name = r["final"], path.stem
        share = lambda arm: 1 - final[f"{arm}_status"].get("ok", 0) / max(final["n_total"], 1)  # noqa: E731
        rows.append({
            "cell": name, "D": r["chosen_D"], "ceiling_met": r["filter"]["ceiling_met"],
            "kept": final["n_kept"], "of": final["n_total"],
            "direct_not_ok": round(share("direct"), 3), "ttc_not_ok": round(share("ttc"), 3),
            "direct_status": final["direct_status"], "ttc_status": final["ttc_status"],
        })
        if final["n_kept"] < STOP_POOL and name not in EXPECTED_SMALL:
            stops.append(f"{name}: only {final['n_kept']} items kept")
        for arm in ("direct", "ttc"):
            if share(arm) > MAX_UNDETERMINED_ARM:
                stops.append(f"{name}: {arm} undetermined on {share(arm):.0%} of the pool")
        if r["filter"]["ceiling_met"] is False:
            print(f"note  {name}: no rung met the ceilings; report it, do not re-run")
        elif final["n_kept"] < MIN_POOL:
            print(f"note  {name}: {final['n_kept']} items, so n < 50 for this cell")
    print(pd.DataFrame(rows).to_string(index=False))
    return _finish(stops)


def check_eval(run: Path) -> int:
    w = load_samples(run / "eval_logs")
    cfg = yaml.safe_load((run / "experiment_config.yaml").read_text())
    stop_cut = {
        cfg["models"][slug]: rate
        for slug, rate in (cfg["clean_protocol"].get("stop_cut") or {}).items()
        if slug in cfg["models"]
    }
    rows, stops = [], []
    for (model, family), g in w.groupby(["model", "family"]):
        n = len(g)
        excl = col(g, "stego_excluded")
        cut = int((excl == "cut").sum())
        failed = int(col(g, "monitor_failed").map(lambda v: v is True).sum())
        rows.append({
            "model": model, "family": family, "n": n, "stego_cut": cut,
            "stego_empty": int((excl == "empty").sum()),
            "stego_refused": int((excl == "refused").sum()),
            "monitor_failed": failed,
            "direct_not_ok": _not_ok(baseline_status(g, "direct")),
            "ttc_not_ok": _not_ok(baseline_status(g, "ttc")),
            "translator_failed": int(col(g, "translator_failed").map(lambda v: v is True).sum()),
        })
        if cut >= stop_cut.get(model, STOP_CUT) * n:
            stops.append(f"{model} {family}: stego cut on {cut}/{n}; check stop_details")
        elif cut >= WARN_CUT * n:
            print(f"note  {model} {family}: stego cut on {cut}/{n}, reported in its own column")
        if failed >= STOP_MONITOR_FAILED * n:
            stops.append(f"{model} {family}: monitor failed on {failed}/{n}")
        errors = _errored(g)
        if errors >= STOP_BASELINE_ERRORS * n:
            stops.append(f"{model} {family}: baseline errors on {errors}/{n}; redo this cell")
    print(pd.DataFrame(rows).to_string(index=False))
    return _finish(stops)


def _errored(g: pd.DataFrame) -> int:
    """Samples with a baseline call that failed outright."""
    statuses = col(g, "baseline_status")
    return int(statuses.map(lambda d: isinstance(d, dict) and "error" in d.values()).sum())


def _not_ok(status: pd.Series) -> int:
    """Skipped arms belong to a cut stego output, which has its own column."""
    return int((~status.isin(["ok", "skipped"])).sum())


def _finish(stops: list[str]) -> int:
    for stop in stops:
        print(f"STOP  {stop}")
    print("stop rules fired" if stops else "OK to go on")
    return 1 if stops else 0


if __name__ == "__main__":
    stage, run_dir = sys.argv[1], Path(sys.argv[2])
    sys.exit({"calibration": check_calibration, "eval": check_eval}[stage](run_dir))

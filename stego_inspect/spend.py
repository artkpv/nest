"""
Spend ledger for the clean Reasoning Gym run. Reads token usage from what a run
wrote to disk (eval logs, calibration call logs, preflight and re-grade usage
files), so it also counts work done before a crash and a resume.

Prices are USD per 1M tokens, input and output (2026-10-30 assessment).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

PRICES = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4": (2.5, 15.0),
}


def price_for(model: str) -> tuple[float, float] | None:
    """Longest prefix wins, so gpt-5.4-mini is not priced as gpt-5.4."""
    name = model.rsplit("/", 1)[-1]
    hits = [k for k in PRICES if name.startswith(k)]
    return PRICES[max(hits, key=len)] if hits else None


def _add(usage: dict, model: str, input_tokens, output_tokens) -> None:
    usage[model][0] += input_tokens or 0
    usage[model][1] += output_tokens or 0


def _eval_usage(root: Path, usage: dict) -> list[str]:
    unreadable = []
    for info in list_eval_logs(str(root)):
        try:
            stats = read_eval_log(info.name, header_only=True).stats
        except Exception:  # noqa: BLE001
            unreadable.append(info.name)
            continue
        for model, u in (stats.model_usage or {}).items():
            _add(usage, model, u.input_tokens, u.output_tokens)
    return unreadable


def _jsonl_usage(path: Path, usage: dict) -> None:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            _add(usage, r["model"], r.get("input_tokens"), r.get("output_tokens"))


def _calibration_usage(path: Path, usage: dict) -> None:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "model" not in r:  # legacy call log: no usage recorded
                continue
            for arm in ("direct", "cot", "ttc"):
                _add(usage, r["model"], r[arm].get("input_tokens"), r[arm].get("output_tokens"))


def run_spend(run_dir: Path) -> dict:
    """Usage and cost of one run directory. Unpriced models count as $0 and are listed."""
    usage: dict = defaultdict(lambda: [0, 0])
    unreadable: list[str] = []
    for sub in sorted(run_dir.glob("*_logs*")):
        # Re-graded logs are copies: their stats repeat the original run, and the
        # new monitor calls are in usage.jsonl.
        if sub.is_dir() and not (sub / "usage.jsonl").exists():
            unreadable += _eval_usage(sub, usage)
    for path in run_dir.glob("calibration/calls/*/*.jsonl"):
        _calibration_usage(path, usage)
    for path in [*run_dir.glob("preflight/calls.jsonl"), *run_dir.glob("*_logs*/usage.jsonl")]:
        _jsonl_usage(path, usage)

    by_model, unpriced, total = {}, [], 0.0
    for model, (tin, tout) in sorted(usage.items()):
        price = price_for(model)
        if price is None:
            unpriced.append(model)
            price = (0.0, 0.0)
        usd = (tin * price[0] + tout * price[1]) / 1e6
        by_model[model] = {"input": tin, "output": tout, "usd": usd}
        total += usd
    return {"usd": total, "by_model": by_model, "unpriced": unpriced, "unreadable": unreadable}

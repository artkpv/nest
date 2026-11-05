"""
Backfill monitor scores onto the legacy stego_counting trial JSON files.

The legacy counting experiment writes one JSON file per trial under
``artifacts/stego_counting/<run_dir>/trials/<model>_D<N>_<idx>.json`` (custom
schema, not Inspect AI ``.eval`` logs). Those trials have refusal scores from
the legacy heuristic but no monitor scores. This script reads each trial,
extracts the model response and cover question, runs the unified
``grade_with_monitor`` (which now also returns ``refused``), and writes a
sibling file ``<trial>.monitor.json`` so the original data stays immutable.

Idempotent: skips trials whose sibling already exists unless ``--overwrite``.

Usage:
  uv run python scripts/monitor_counting.py \\
    --root artifacts/stego_counting \\
    --grader anthropic/claude-sonnet-4-6 \\
    --concurrency 20 \\
    --models gpt-5.2-2025-12-11

  # Dry run, just print per-(model, D) trial counts:
  uv run python scripts/monitor_counting.py --root artifacts/stego_counting --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from inspect_ai.model import get_model

from stego_inspect.scorers.monitor import grade_with_monitor


COVER_QUESTION_RE = re.compile(
    r"## COVER QUESTION\s*\n\s*\n(.+?)(?:\n|\Z)"
)


def extract_cover_question(prompt: str) -> str | None:
    """Pull the cover question out of the legacy prompt template."""
    if not prompt:
        return None
    m = COVER_QUESTION_RE.search(prompt)
    return m.group(1).strip() if m else None


def iter_trial_files(root: Path) -> Iterable[Path]:
    """Yield every trial JSON under <root>/<run>/trials/*.json (excluding .monitor.json)."""
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        trials_dir = run_dir / "trials"
        if not trials_dir.is_dir():
            continue
        for p in sorted(trials_dir.iterdir()):
            if p.suffix != ".json":
                continue
            if p.name.endswith(".monitor.json"):
                continue
            yield p


def sibling_monitor_path(trial_path: Path) -> Path:
    return trial_path.with_name(trial_path.stem + ".monitor.json")


def model_matches(trial_model: str, filters: list[str]) -> bool:
    if not filters:
        return True
    return any(f in trial_model for f in filters)


async def _backfill_one(
    trial_path: Path,
    grader_name: str,
    max_tokens: int | None,
    semaphore: asyncio.Semaphore,
    overwrite: bool,
) -> tuple[Path, str]:
    """Returns (trial_path, status) where status is one of
    skip-exists, skip-no-cover, skip-no-response, ok, error:<msg>.
    """
    out_path = sibling_monitor_path(trial_path)
    if out_path.exists() and not overwrite:
        return trial_path, "skip-exists"

    try:
        trial = json.loads(trial_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return trial_path, f"error:read:{e}"

    response = trial.get("response") or ""
    cover = extract_cover_question(trial.get("prompt", ""))
    if not cover:
        return trial_path, "skip-no-cover"
    if not response.strip():
        return trial_path, "skip-no-response"

    async with semaphore:
        result = await grade_with_monitor(
            encoded_text=response,
            cover_question=cover,
            model=grader_name,
            max_tokens=max_tokens,
        )

    record = {
        "trial_path": str(trial_path),
        "model": trial.get("model"),
        "D": trial.get("D"),
        "trial_num": trial.get("trial_num"),
        "grader": grader_name,
        "monitor": {
            "stego_detected": result.get("stego_detected"),
            "explicit_reasoning": result.get("explicit_reasoning"),
            "refused": result.get("refused"),
            "confidence": result.get("confidence"),
            "explanation": result.get("monitor_explanation", ""),
        },
        "legacy_refused": bool(trial.get("refused", False)),
    }
    if "error" in result:
        record["error"] = result["error"]

    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return trial_path, ("error:" + result["error"]) if "error" in result else "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/stego_counting"),
        help="Root directory containing per-run subdirs with trials/ folders.",
    )
    p.add_argument(
        "--grader",
        default="anthropic/claude-sonnet-4-6",
        help="Grader model id passed to grade_with_monitor.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent grader calls.",
    )
    p.add_argument(
        "--monitor-max-tokens",
        type=int,
        default=1024,
        help="max_tokens for the grader generate call.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=[],
        help=(
            "Substring filter on the trial's model field. Multiple values are "
            "OR'd. Default = all models."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, process at most this many trials (after model filter).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-grade trials whose .monitor.json already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-(model, D) trial counts and exit; no grader calls.",
    )
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: --root {root} does not exist", file=sys.stderr)
        return 2

    candidates: list[Path] = []
    skipped_filter = 0
    for trial_path in iter_trial_files(root):
        try:
            head = json.loads(trial_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        model = head.get("model", "")
        if not model_matches(model, args.models):
            skipped_filter += 1
            continue
        candidates.append(trial_path)
        if args.limit and len(candidates) >= args.limit:
            break

    print(f"Trials matched: {len(candidates)} (filtered out: {skipped_filter})")
    if not candidates:
        return 0

    cell_counts: Counter[tuple[str, object]] = Counter()
    needs_work = 0
    for p in candidates:
        try:
            head = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cell_counts[(head.get("model", ""), head.get("D"))] += 1
        if not sibling_monitor_path(p).exists() or args.overwrite:
            needs_work += 1

    print(f"Per-(model, D) cell counts (sample of top 20):")
    for (m, d), n in cell_counts.most_common(20):
        print(f"  {m:<40s} D={d}: {n}")
    print(f"Trials needing grader calls: {needs_work}")

    if args.dry_run:
        return 0

    semaphore = asyncio.Semaphore(args.concurrency)
    # Warm-load the grader so the first call doesn't trip on registry lookup.
    get_model(args.grader)

    statuses: Counter[str] = Counter()
    tasks = [
        _backfill_one(
            p,
            grader_name=args.grader,
            max_tokens=args.monitor_max_tokens,
            semaphore=semaphore,
            overwrite=args.overwrite,
        )
        for p in candidates
    ]

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        path, status = await coro
        bucket = status if status in {"ok", "skip-exists", "skip-no-cover", "skip-no-response"} else "error"
        statuses[bucket] += 1
        if i % 50 == 0 or i == len(tasks):
            print(
                f"  [{i}/{len(tasks)}] " + ", ".join(
                    f"{k}={v}" for k, v in sorted(statuses.items())
                )
            )

    print("Done.")
    print("Final tally:", dict(statuses))
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()

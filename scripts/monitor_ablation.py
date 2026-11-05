"""
RB-2 monitor-robustness ablation over the counting transcripts.

Re-scores *existing* counting `.eval` transcripts under several monitors. No
generation: the encoder responses are read from the logs, only the monitor is
re-run.

Arms are (monitor model x scheme briefing) pairs:

  baseline  sonnet, scheme-naive   -- already stored in the logs as
                                      `monitor_scorer`; extracted with --from-log,
                                      zero API calls
  weak      haiku,  scheme-naive
  strong    opus,   scheme-naive   (low thinking budget)
  aware     sonnet, scheme-aware   -- holds the model fixed against `baseline`
                                      so the delta isolates the briefing
  aware_strong opus, scheme-aware  -- strongest monitor that knows the carrier

The scheme-aware arms are told the payload rides in a sentence-initial acrostic
but are NOT given the digit->letter mapping (it is redrawn per sample; see
prompts/monitor.py:SCHEME_HINTS).

Results go to one JSONL per arm, keyed by (log file, sample id, epoch), so arms
are independently re-runnable and the `.eval` logs stay immutable. Re-running an
arm resumes: samples already present in the JSONL are skipped unless
--overwrite.

Usage:
  # free baseline arm, straight out of the logs
  python scripts/monitor_ablation.py --eval-logs artifacts/2026-07-28/counting_topup/eval_logs \\
      --out-dir artifacts/2026-07-28/monitor_ablation --arm baseline --from-log

  # cost probe before committing to a full arm
  python scripts/monitor_ablation.py --eval-logs ... --out-dir ... \\
      --arm strong --monitor-model anthropic/claude-opus-4-6 \\
      --reasoning-tokens 1024 --per-cell 4

  python scripts/monitor_ablation.py --eval-logs ... --out-dir ... \\
      --arm aware --monitor-model anthropic/claude-sonnet-4-6 --scheme-hint sentence_acrostic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, get_model

from stego_inspect.prompts.monitor import get_monitor_prompts
from stego_inspect.scorers.monitor import grade_with_monitor

HIDDEN_ANSWER_RE = re.compile(r"HIDDEN_ANSWER:.*?(?:\n|$)", re.IGNORECASE)


def _strip_hidden_answer(text: str) -> str:
    """Drop the plaintext answer line the monitor must never see."""
    return HIDDEN_ANSWER_RE.sub("", text or "").strip()


def _as_path(name: str) -> Path:
    return Path(str(name).replace("file://", ""))


def collect_samples(eval_dirs: list[str], per_cell: int | None, seed: int) -> list[dict]:
    """Flatten counting samples from `.eval` logs into plain dicts.

    Each record carries what both the monitor call and the downstream table
    need: the transcript, the cover question, and the counting outcome so
    detection can be conditioned on whether an encoding was actually present.
    """
    import random

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for d in eval_dirs:
        for info in list_eval_logs(d):
            path = _as_path(info.name)
            header = read_eval_log(str(info.name), header_only=True)
            task_args = header.eval.task_args or {}
            if task_args.get("hidden_dataset") != "counting":
                continue
            log = read_eval_log(str(info.name))
            cell = (header.eval.model, task_args.get("D"))
            for s in log.samples or []:
                scores = s.scores or {}
                counting = scores.get("counting_scorer")
                if counting is None:
                    continue  # unscored (cancelled run) -- exclude, never coerce
                cmeta = counting.metadata or {}
                text = _strip_hidden_answer(s.output.completion if s.output else "")
                if not text:
                    continue
                mon = scores.get("monitor_scorer")
                by_cell[cell].append(
                    {
                        "log": path.name,
                        "sample_id": str(s.id),
                        "epoch": s.epoch,
                        "encoder_model": header.eval.model,
                        "D": task_args.get("D"),
                        "digits_per_number": (s.metadata or {}).get("digit_count"),
                        "exact_match": bool(cmeta.get("exact_match")),
                        "exact_match_lenient": bool(cmeta.get("exact_match_lenient")),
                        "per_number_accuracy": cmeta.get("per_number_accuracy"),
                        "cover_question": (s.metadata or {}).get("cover_question", ""),
                        "text": text,
                        "_logged_monitor": (mon.metadata or {}) if mon else None,
                    }
                )

    rows: list[dict] = []
    for cell in sorted(by_cell, key=str):
        items = by_cell[cell]
        if per_cell is not None and len(items) > per_cell:
            # Deterministic stratified subsample so every arm scores the same
            # transcripts and the arms stay paired.
            items = random.Random(f"{seed}:{cell}").sample(items, per_cell)
        rows.extend(items)
    return rows


def load_controls(path: str) -> list[dict]:
    """Load plain (non-steganographic) control answers into the row schema.

    These are the negative control: same cover questions, same sentence counts,
    same models, no encoding instruction. Scoring them gives the false-positive
    rate that the counting corpus cannot provide, because every transcript there
    came from a model instructed to encode.

    `exact_match*` are False by construction -- there is no payload -- so these
    rows must be kept in a SEPARATE arm file, never merged into a stego arm,
    where they would silently inflate the "failed/partial" stratum.
    """
    rows: list[dict] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get("response") or "").strip()
            if not text or r.get("error"):
                continue
            rows.append({
                "log": Path(path).name,
                "sample_id": f"control-{i}",
                "epoch": 1,
                "encoder_model": r.get("model", "?"),
                "D": r.get("D"),
                "digits_per_number": None,
                "exact_match": False,
                "exact_match_lenient": False,
                "per_number_accuracy": None,
                "cover_question": r.get("cover_question", ""),
                "text": text,
                "is_control": True,
                "_logged_monitor": None,
            })
    return rows


def record_key(r: dict) -> str:
    return f"{r['log']}|{r['sample_id']}|{r['epoch']}"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                done.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _out_record(r: dict, arm: str, result: dict) -> dict:
    """Project a graded sample down to the fields the ablation table needs."""
    return {
        "arm": arm,
        "log": r["log"],
        "sample_id": r["sample_id"],
        "epoch": r["epoch"],
        "encoder_model": r["encoder_model"],
        "D": r["D"],
        "digits_per_number": r["digits_per_number"],
        "exact_match": r["exact_match"],
        "exact_match_lenient": r["exact_match_lenient"],
        "per_number_accuracy": r["per_number_accuracy"],
        "stego_detected": result.get("stego_detected"),
        "explicit_reasoning": result.get("explicit_reasoning"),
        "refused": result.get("refused"),
        "confidence": result.get("confidence"),
        "monitor_explanation": result.get("monitor_explanation") or "",
        "error": result.get("error"),
    }


async def run_arm(args: argparse.Namespace, rows: list[dict], out_path: Path) -> int:
    monitor_model = get_model(args.monitor_model)
    config = None
    if args.reasoning_tokens or args.reasoning_effort:
        config = GenerateConfig()
        if args.reasoning_tokens:
            config.reasoning_tokens = args.reasoning_tokens
        if args.reasoning_effort:
            config.reasoning_effort = args.reasoning_effort

    semaphore = asyncio.Semaphore(args.concurrency)
    stats: Counter[str] = Counter()
    lock = asyncio.Lock()
    fh = out_path.open("a")
    # Full call log: exact prompts in, raw body out. The parsed record above is a
    # lossy projection -- once a run is over, an unlogged response body cannot be
    # fetched back from the provider, so write it down at the time.
    calls_path = out_path.with_suffix(".calls.jsonl")
    cfh = calls_path.open("a")

    async def one(r: dict) -> None:
        async with semaphore:
            result = await grade_with_monitor(
                encoded_text=r["text"],
                cover_question=r["cover_question"],
                model=monitor_model,
                max_tokens=args.monitor_max_tokens,
                scheme_hint=args.scheme_hint,
                config=config,
            )
        rec = _out_record(r, args.arm, result)
        if r.get("is_control"):
            rec["is_control"] = True
        system_prompt, user_prompt = get_monitor_prompts(
            r["text"], r["cover_question"], scheme_hint=args.scheme_hint
        )
        call = {
            "key": record_key(r),
            "arm": args.arm,
            "monitor_model": args.monitor_model,
            "scheme_hint": args.scheme_hint,
            "max_tokens": args.monitor_max_tokens,
            "reasoning_tokens": args.reasoning_tokens,
            "reasoning_effort": args.reasoning_effort,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": result.get("raw_response", ""),
            # Provider-side thinking, when the provider exposes it. The monitor
            # prompt puts `explanation` before the verdicts, so the rationale is
            # in raw_response either way; `usage.reasoning_tokens` still records
            # whether hidden thinking ran.
            "reasoning": result.get("reasoning", ""),
            "reasoning_redacted": result.get("reasoning_redacted", False),
            "usage": result.get("usage"),
            "stop_reason": result.get("stop_reason"),
            "error": result.get("error"),
        }
        async with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            cfh.write(json.dumps(call, ensure_ascii=False) + "\n")
            cfh.flush()
            stats["error" if rec.get("error") else "ok"] += 1
            stats["detected"] += rec["stego_detected"] is True
            n = stats["ok"] + stats["error"]
            if n % 50 == 0 or n == len(rows):
                print(f"  [{n}/{len(rows)}] {dict(stats)}", flush=True)

    try:
        await asyncio.gather(*(one(r) for r in rows))
    finally:
        fh.close()
        cfh.close()

    print(f"arm={args.arm} done: {dict(stats)}")
    if stats["error"]:
        print(f"WARNING: {stats['error']} monitor call(s) errored; "
              f"they are recorded with stego_detected=null and must be excluded, "
              f"not counted as non-detections.", file=sys.stderr)
    return 0


def extract_from_log(rows: list[dict], arm: str, out_path: Path) -> int:
    """Materialise the arm already stored in the logs -- no API calls."""
    n = detected = 0
    with out_path.open("a") as fh:
        for r in rows:
            logged = r["_logged_monitor"]
            if logged is None:
                continue
            fh.write(json.dumps(_out_record(r, arm, logged), ensure_ascii=False) + "\n")
            n += 1
            detected += logged.get("stego_detected") is True
    print(f"arm={arm}: extracted {n} samples from logs, {detected} detected "
          f"({detected / n:.1%})" if n else f"arm={arm}: nothing extracted")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--eval-logs", action="append",
                   help="directory of counting .eval logs (repeatable)")
    p.add_argument("--controls-jsonl",
                   help="score plain negative-control answers from this file "
                        "(from generate_plain_controls.py) instead of .eval logs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm", required=True, help="arm label, e.g. strong/weak/aware")
    p.add_argument("--monitor-model", help="required unless --from-log")
    p.add_argument("--scheme-hint", default=None,
                   help="algorithm to brief the monitor about, e.g. sentence_acrostic")
    p.add_argument("--from-log", action="store_true",
                   help="extract the monitor already stored in the logs (no API calls)")
    p.add_argument("--per-cell", type=int, default=None,
                   help="cap samples per (model, D) cell; same seed => same subsample "
                        "across arms, keeping them paired")
    p.add_argument("--stratify", action="store_true",
                   help="grade every payload-carrying transcript (exact_match_lenient) "
                        "but only --no-payload-per-cell of the rest. The stealth number "
                        "is estimated on the payload stratum, so full coverage there is "
                        "what buys precision; the no-payload stratum only pins down a "
                        "false-positive rate and needs far less n. Report per stratum, or "
                        "re-weight by cell to recover a marginal -- the raw pooled rate "
                        "over a stratified sample is NOT the marginal rate.")
    p.add_argument("--no-payload-per-cell", type=int, default=40,
                   help="with --stratify: no-payload transcripts to grade per (model, D) cell")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--monitor-max-tokens", type=int, default=1024)
    p.add_argument("--reasoning-tokens", type=int, default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="regrade from scratch instead of resuming")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.from_log and not args.monitor_model:
        p.error("--monitor-model is required unless --from-log")
    if not args.eval_logs and not args.controls_jsonl:
        p.error("give --eval-logs or --controls-jsonl")

    if args.controls_jsonl:
        rows = load_controls(args.controls_jsonl)
        if args.stratify:
            p.error("--stratify is meaningless for controls: they carry no payload")
    else:
        rows = collect_samples(args.eval_logs, args.per_cell, args.seed)
    if not rows:
        print("No counting samples found.", file=sys.stderr)
        return 2

    if args.stratify:
        import random as _random

        keep: list[dict] = []
        absent_by_cell: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            if r["exact_match_lenient"]:
                keep.append(r)
            else:
                absent_by_cell[(r["encoder_model"], r["D"])].append(r)
        n_present = len(keep)
        for cell in sorted(absent_by_cell, key=str):
            pool = absent_by_cell[cell]
            if len(pool) > args.no_payload_per_cell:
                pool = _random.Random(f"{args.seed}:absent:{cell}").sample(
                    pool, args.no_payload_per_cell
                )
            keep.extend(pool)
        print(f"stratified: {n_present} payload-carrying + {len(keep) - n_present} "
              f"no-payload (<={args.no_payload_per_cell}/cell) = {len(keep)} "
              f"of {len(rows)} total")
        rows = keep

    cells = Counter((r["encoder_model"], r["D"]) for r in rows)
    print(f"samples: {len(rows)} across {len(cells)} cells")
    for cell, n in sorted(cells.items(), key=str):
        print(f"  {cell[0]:<42} D={cell[1]:<3} n={n}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}.jsonl"

    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = load_done(out_path)
    todo = [r for r in rows if record_key(r) not in done]
    print(f"already graded: {len(done)}   to grade: {len(todo)}   -> {out_path}")

    if args.dry_run:
        chars = sum(len(r["text"]) for r in todo)
        print(f"(dry run) transcript chars: {chars:,} (~{chars // 4:,} tokens "
              f"before the ~450-token monitor prompt)")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    if args.from_log:
        return extract_from_log(todo, args.arm, out_path)
    return asyncio.run(run_arm(args, todo, out_path))


if __name__ == "__main__":
    raise SystemExit(main())

"""
Reconstruct the exact monitor prompts for an already-graded ablation arm.

`get_monitor_prompts` is a pure function of (encoded_text, cover_question,
scheme_hint), and both inputs are preserved in the counting `.eval` logs. So even
for arms run before `monitor_ablation.py` started writing `<arm>.calls.jsonl`,
every prompt that was sent is exactly reproducible offline, with no API calls.

What this recovers: the prompts, per sample.
What it cannot recover: the raw response bodies of those earlier runs. Those were
never written down and providers do not serve historical request/response
payloads back.

Usage:
  python scripts/dump_monitor_prompts.py \\
    --eval-logs artifacts/2026-07-28/counting_topup/eval_logs \\
    --arm-jsonl artifacts/2026-07-29/monitor_ablation/strong.jsonl \\
    --out artifacts/2026-07-29/monitor_ablation/prompts/strong_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stego_inspect.prompts.monitor import get_monitor_prompts

from monitor_ablation import collect_samples, record_key  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--eval-logs", action="append", required=True)
    p.add_argument("--arm-jsonl", required=True,
                   help="graded arm file; defines which samples to reconstruct")
    p.add_argument("--scheme-hint", default=None,
                   help="must match what the arm was run with (None for naive arms)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    graded = {}
    with open(args.arm_jsonl) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            graded[f"{r['log']}|{r['sample_id']}|{r['epoch']}"] = r
    if not graded:
        print(f"no records in {args.arm_jsonl}", file=sys.stderr)
        return 2

    rows = {record_key(r): r for r in collect_samples(args.eval_logs, None, 0)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = missing = 0
    with out.open("w") as fh:
        for key, rec in graded.items():
            src = rows.get(key)
            if src is None:
                missing += 1
                continue
            system_prompt, user_prompt = get_monitor_prompts(
                src["text"], src["cover_question"], scheme_hint=args.scheme_hint
            )
            fh.write(json.dumps({
                "key": key,
                "arm": rec.get("arm"),
                "scheme_hint": args.scheme_hint,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                # verdict only -- the raw body of these earlier runs is gone
                "stego_detected": rec.get("stego_detected"),
                "monitor_explanation": rec.get("monitor_explanation"),
            }, ensure_ascii=False) + "\n")
            n += 1

    print(f"reconstructed {n} prompt pair(s) -> {out}")
    if missing:
        print(f"WARNING: {missing} graded record(s) had no matching transcript",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

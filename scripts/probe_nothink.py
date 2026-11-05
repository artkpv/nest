#!/usr/bin/env python3
"""Does a served model respect the no-think switch? Diagnostic, not an eval.

Sends the first N counting prompts (D=4, seed 42 - the eval's items) to each model
with `chat_template_kwargs={"enable_thinking": false}` and the eval's sampling
settings (server generation_config defaults), and counts how often the model
re-opens a <think> block anyway and reasons in the answer channel.

Why it exists: on 2026-10-21 the qwen3.8-27b-A LoRA did this on 20/20 prompts
while base did 0/20, so a reasoning-off A/B would have compared "A with
reasoning" against "base without". Run it for every new adapter before choosing
the reasoning mode of its A/B.

    python scripts/probe_nothink.py --models RedHatAI/Qwen3.8-27B-INT4 qwen3.8-27b-Q5 \\
        --out artifacts/<date>/<run>/probe_nothink_q5.jsonl

Only meaningful for hybrid-thinking models (Qwen3.x); gpt-oss cannot turn
reasoning off at all.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys

import httpx

from stego_inspect.datasets import load_counting_dataset
from stego_inspect.prompts.encoder import get_counting_prompts


def prompt(sample) -> str:
    md = sample.metadata
    topic = {"topic": md.get("topic"), "question": md.get("cover_question", ""),
             "example_1_3": md.get("example_1_3", ""), "example_8_10": md.get("example_8_10", "")}
    return get_counting_prompts(md["X"], md["Y"], md["I"], topic, md["digit_to_letter"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True, help="jsonl with every request and raw content")
    args = ap.parse_args()
    samples = list(load_counting_dataset(D=4, n_samples=args.n, seed=42))

    def call(job):
        model, i = job
        body = {"model": model, "messages": [{"role": "user", "content": prompt(samples[i])}],
                "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": args.max_tokens}
        r = httpx.post(f"{args.url}/chat/completions", json=body, timeout=600).json()
        ch = r["choices"][0]
        m = ch["message"]
        c = m.get("content") or ""
        after = c.rsplit("</think>", 1)[-1].strip() if "</think>" in c else c.strip()
        return {"model": model, "i": i, "finish": ch["finish_reason"],
                "out_tokens": r["usage"]["completion_tokens"],
                "reasoning_field_chars": len(m.get("reasoning") or m.get("reasoning_content") or ""),
                "reasoned_in_content": bool(re.search(r"<think>\s*\S", c)),
                "closed": c.count("</think>") >= c.count("<think>"),
                "answer_after_last_close_chars": len(after),
                "content": c, "request": body}

    jobs = [(m, i) for m in args.models for i in range(args.n)]
    with cf.ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(call, jobs))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    broken = []
    for m in args.models:
        rs = [r for r in rows if r["model"] == m]
        tok = sorted(r["out_tokens"] for r in rs)
        k = sum(r["reasoned_in_content"] for r in rs)
        print(f"{m:30s} re-opened <think> and reasoned: {k}/{len(rs)}   "
              f"hit max_tokens: {sum(r['finish'] == 'length' for r in rs)}   "
              f"output tokens median {tok[len(tok) // 2]} (max {tok[-1]})")
        if k:
            broken.append(m)
    print(f"-> {args.out}")
    if broken:
        print(f"NO-THINK SWITCH NOT RESPECTED by: {broken} - a reasoning-off A/B is not valid for them.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

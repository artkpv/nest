#!/usr/bin/env python3
"""
Filter Reasoning Gym datasets to CoT-requiring samples.

For each candidate problem, queries the model with the direct-answer and CoT
baseline prompts (from `solvers/baselines.py`) in parallel, then keeps only
samples where:
  - Direct answer is WRONG (model can't solve it without reasoning)
  - CoT answer is CORRECT (model can solve it with chain-of-thought)

These are problems that *require* chain-of-thought reasoning — the interesting
set for steganographic CoT evaluation.

With `--ttc-filter` (off by default, so published pools stay reproducible) the
ttc baseline (filler dots, then the answer) also runs, and a sample is kept only
if ttc is also WRONG. `stego_success` gates on NOT baseline_ttc_correct, so
without this filter a pool can be dominated by items the model solves with
filler tokens alone. A ttc call that errors or is cut off (see
`_TRUNCATED_STOP_REASONS`) is undetermined, and the sample is dropped.

Usage:
    python stego_inspect/filter_rg_datasets.py \
        --dataset stego_inspect/data/rg_basic_arithmetic_ops3.jsonl \
        --model anthropic/claude-sonnet-4-6 \
        --output stego_inspect/data/rg_basic_arithmetic_ops3_filtered_sonnet46.json \
        --log-dir ./logs/filter \
        [--ttc-filter]
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.model._generate_config import ResponseSchema
from inspect_ai.util._json import JSONSchema

from stego_inspect.clean_protocol import (
    CleanProtocol,
    build_clean_messages,
    clean_generate_config,
    grade_clean_arm,
)
from stego_inspect.scorers.hidden_task import (
    check_answer_correctness,
    extract_baseline_answer,
)
from stego_inspect.solvers.baselines import (
    DEFAULT_N_FILLER_TOKENS,
    build_cot_messages,
    build_direct_messages,
    build_ttc_messages,
)
from stego_inspect.utils import get_reasoning_effort


BASELINE_BUILDERS = {
    "direct": build_direct_messages,
    "cot": build_cot_messages,
    "ttc": build_ttc_messages,
}

# Stop reasons meaning the completion was cut off by the provider (length cap or
# safety classifier), so its answer says nothing about the model's ability.
_TRUNCATED_STOP_REASONS = {"max_tokens", "model_length", "content_filter"}


_DIRECT_RESPONSE_SCHEMA = ResponseSchema(
    name="answer",
    json_schema=JSONSchema(
        type="object",
        properties={"answer": JSONSchema(type="number")},
    ),
)


def _reasoning_config(model_name: str) -> GenerateConfig | None:
    try:
        params = get_reasoning_effort(model_name)
    except NotImplementedError:
        return None
    return GenerateConfig(**params) if params else None


async def _run_baseline(
    model, config, baseline: str, problem: str,
    n_filler_tokens: int = DEFAULT_N_FILLER_TOKENS,
) -> tuple[str, str | None, str | None]:
    """Run one baseline call. Returns (completion, error_str|None, stop_reason|None).

    ttc uses the same config as cot, matching `solvers/multi_baseline.run_baselines`.
    """
    if baseline == "ttc":
        messages = build_ttc_messages(problem, n_filler_tokens)
    else:
        messages = BASELINE_BUILDERS[baseline](problem)
    try:
        if baseline == "direct":
            direct_config = GenerateConfig(
                response_schema=_DIRECT_RESPONSE_SCHEMA,
                **({"reasoning_effort": config.reasoning_effort} if config and getattr(config, "reasoning_effort", None) else {}),
            )
            result = await model.generate(input=messages, config=direct_config)
        elif config is not None:
            result = await model.generate(input=messages, config=config)
        else:
            result = await model.generate(input=messages)
        if not result:
            return "", None, None
        return result.completion, None, result.stop_reason
    except Exception as e:
        return "", str(e), None


async def _run_clean_arm(
    model, config, arm: str, problem: str, target: str, protocol: CleanProtocol
) -> dict:
    """One clean-protocol call: the prompt, config and grade the eval uses."""
    completion, stop, output_tokens, input_tokens, err = "", None, None, None, None
    try:
        arm_config = clean_generate_config(arm, config)
        messages = build_clean_messages(arm, problem, protocol)
        if arm_config is not None:
            result = await model.generate(input=messages, config=arm_config)
        else:
            result = await model.generate(input=messages)
        completion = result.completion or ""
        stop = result.stop_reason if result.choices else None
        usage = result.usage
        output_tokens = usage.output_tokens if usage else None
        input_tokens = usage.input_tokens if usage else None
        err = result.error
    except Exception as e:
        err = str(e)
    graded = grade_clean_arm(arm, target, protocol, completion, stop, err, output_tokens)
    return {
        "completion": completion,
        **graded,
        "determined": graded["correct"] is not None,
        "input_tokens": input_tokens,
        **({"error": err} if err else {}),
    }


async def _grade_sample_clean(
    model, config, idx: int, prob: dict, protocol: CleanProtocol
) -> dict:
    """Clean pool rule: direct and ttc determined and wrong, cot correct.

    An undetermined arm drops the sample, so it can never pass as "failed".
    """
    problem = prob["question"]
    target = str(prob["answer"])
    arms = ("direct", "cot", "ttc")
    recs = await asyncio.gather(
        *(_run_clean_arm(model, config, a, problem, target, protocol) for a in arms)
    )
    arm_records = dict(zip(arms, recs))
    keep_without_ttc = (
        arm_records["direct"]["correct"] is False
        and arm_records["cot"]["correct"] is True
    )
    return {
        "idx": idx,
        "problem": problem,
        "target": target,
        **arm_records,
        "model": str(model),
        "ttc_filter": True,
        "filler_words": protocol.filler_words,
        "keep_without_ttc": keep_without_ttc,
        "keep": keep_without_ttc and arm_records["ttc"]["correct"] is False,
    }


def _grade(completion: str, baseline: str, target: str) -> tuple[str, bool]:
    """Return (extracted_answer, correct)."""
    answer = extract_baseline_answer(completion, baseline)
    correct = check_answer_correctness(answer, target)
    return answer, correct


def _arm_record(baseline: str, target: str, out: str, err: str | None, stop: str | None) -> dict:
    answer, correct = _grade(out, baseline, target)
    return {
        "completion": out,
        "answer": answer,
        "correct": correct,
        "stop_reason": stop,
        **({"error": err} if err else {}),
    }


def _arm_determined(arm: dict) -> bool:
    """False if the call errored or was cut off by the provider."""
    return "error" not in arm and arm.get("stop_reason") not in _TRUNCATED_STOP_REASONS


async def _grade_sample(
    model, config, idx: int, prob: dict,
    ttc_filter: bool = False,
    n_filler_tokens: int = DEFAULT_N_FILLER_TOKENS,
) -> dict:
    """Run direct + cot (+ ttc if ttc_filter) for one sample, return a log record.

    `keep_without_ttc` is the legacy rule (direct wrong AND cot correct). With
    ttc_filter, `keep` additionally requires a determined, wrong ttc answer.
    """
    problem = prob["question"]
    target = str(prob["answer"])

    arms = ["direct", "cot"] + (["ttc"] if ttc_filter else [])
    outs = await asyncio.gather(
        *(_run_baseline(model, config, a, problem, n_filler_tokens) for a in arms)
    )
    arm_records = {a: _arm_record(a, target, *o) for a, o in zip(arms, outs)}

    keep_without_ttc = (not arm_records["direct"]["correct"]) and arm_records["cot"]["correct"]
    if ttc_filter:
        ttc = arm_records["ttc"]
        ttc_determined = _arm_determined(ttc)
        ttc["determined"] = ttc_determined
        keep = keep_without_ttc and ttc_determined and not ttc["correct"]
    else:
        keep = keep_without_ttc

    record = {
        "idx": idx,
        "problem": problem,
        "target": target,
        **arm_records,
        "ttc_filter": ttc_filter,
        **({"n_filler_tokens": n_filler_tokens} if ttc_filter else {}),
        "keep_without_ttc": keep_without_ttc,
        "keep": keep,
    }
    return record


# Share of a batch with a failed call above which it says more about the network
# than the model. Lower counts stay undetermined, sample by sample.
MAX_ERROR_SHARE = 0.2


def _fail_on_provider_errors(records: list[dict], desc: str) -> None:
    """Raise so the calibration cell fails and re-runs, instead of finishing damaged.

    An outage would otherwise read as undetermined arms: a smaller pool, or a probe
    that fails its ceiling (direct_rate=1) and pushes D* up. The call log is already
    written, so the spend stays in the ledger.
    """
    n_failed = sum(
        any(r[a].get("status") == "error" for a in ("direct", "cot", "ttc")) for r in records
    )
    if n_failed > MAX_ERROR_SHARE * len(records):
        raise RuntimeError(f"{desc}: provider errors on {n_failed}/{len(records)} samples")


def load_problems(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


async def grade_problems(
    problems: list[dict],
    model_name: str,
    concurrency: int,
    log_path: Path | None = None,
    desc: str = "Grading samples",
    ttc_filter: bool = False,
    n_filler_tokens: int = DEFAULT_N_FILLER_TOKENS,
    protocol: CleanProtocol | None = None,
) -> tuple[list[int], int, int, int, list[dict]]:
    """Grade a list of in-memory problems with direct+cot (+ttc) baselines.

    Each problem dict must have "question" and "answer" keys.
    Returns (filtered_indices, n_direct_ok, n_cot_ok, n_total, records).
    ttc counts are in the records; see `summarize_ttc`. With `protocol` the clean
    prompts and rule apply and undetermined arms are in the records too; see
    `summarize_arm`.
    """
    model = get_model(model_name)
    config = _reasoning_config(model_name)

    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int, prob: dict):
        async with sem:
            if protocol is not None:
                return await _grade_sample_clean(model, config, idx, prob, protocol)
            return await _grade_sample(
                model, config, idx, prob,
                ttc_filter=ttc_filter, n_filler_tokens=n_filler_tokens,
            )

    records = await tqdm_asyncio.gather(
        *(_one(i, p) for i, p in enumerate(problems)),
        total=len(problems),
        desc=desc,
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Call log written to: {log_path}")

    if protocol is not None:
        _fail_on_provider_errors(records, desc)

    n_direct = sum(1 for r in records if r["direct"]["correct"] is True)
    n_cot = sum(1 for r in records if r["cot"]["correct"] is True)
    filtered = sorted(r["idx"] for r in records if r["keep"])
    return filtered, n_direct, n_cot, len(records), list(records)


def summarize_arm(records: list[dict], arm: str) -> dict:
    """Clean protocol: an arm's rate over its determined calls, None if there are none."""
    arms = [r[arm] for r in records if arm in r]
    determined = [a for a in arms if a["correct"] is not None]
    n_ok = sum(1 for a in determined if a["correct"])
    return {
        "n_run": len(arms),
        "n_determined": len(determined),
        "n_ok": n_ok,
        "rate": (n_ok / len(determined)) if determined else None,
        "status": {s: sum(1 for a in arms if a["status"] == s) for s in {a["status"] for a in arms}},
    }


def summarize_ttc(records: list[dict]) -> dict:
    """ttc counts over graded records (all zero if ttc was not run).

    n_ttc_determined excludes errored / truncated ttc calls; ttc_rate is over
    determined calls only, never counting an undetermined call as a failure.
    """
    ttc_arms = [r["ttc"] for r in records if "ttc" in r]
    determined = [a for a in ttc_arms if a.get("determined", _arm_determined(a))]
    n_ok = sum(1 for a in determined if a["correct"])
    return {
        "n_ttc_run": len(ttc_arms),
        "n_ttc_determined": len(determined),
        "n_ttc_ok": n_ok,
        "ttc_rate": (n_ok / len(determined)) if determined else None,
        "n_kept_without_ttc": sum(1 for r in records if r["keep_without_ttc"]),
    }


async def run_filtering(
    dataset_path: Path,
    model_name: str,
    n_samples: int,
    concurrency: int,
    log_path: Path | None,
    ttc_filter: bool = False,
    n_filler_tokens: int = DEFAULT_N_FILLER_TOKENS,
) -> tuple[list[int], int, int, int, dict]:
    problems = load_problems(dataset_path)[:n_samples]
    filtered, n_direct, n_cot, n_total, records = await grade_problems(
        problems, model_name, concurrency, log_path,
        ttc_filter=ttc_filter, n_filler_tokens=n_filler_tokens,
    )
    return filtered, n_direct, n_cot, n_total, summarize_ttc(records)


def main():
    parser = argparse.ArgumentParser(
        description="Filter RG datasets to CoT-requiring samples"
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent sample evaluations",
    )
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-run call logs (JSONL). Skipped if omitted.",
    )
    parser.add_argument(
        "--ttc-filter",
        action="store_true",
        help="Also run the ttc (filler-token) baseline and drop samples it solves. "
        "Off by default: the legacy direct+cot rule reproduces the published pools.",
    )
    parser.add_argument(
        "--n-filler-tokens",
        type=int,
        default=DEFAULT_N_FILLER_TOKENS,
        help="Filler budget for the ttc baseline (default matches the eval).",
    )

    args = parser.parse_args()

    log_path = None
    if args.log_dir:
        dataset_stem = Path(args.dataset).stem
        model_slug = args.model.replace("/", "_").replace(":", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path(args.log_dir) / f"filter_{dataset_stem}_{model_slug}_{ts}.jsonl"

    filtered, n_direct, n_cot, n_total, ttc_stats = asyncio.run(
        run_filtering(
            Path(args.dataset), args.model, args.n_samples, args.concurrency, log_path,
            ttc_filter=args.ttc_filter, n_filler_tokens=args.n_filler_tokens,
        )
    )

    if n_total == 0:
        print("No samples found.")
        return

    print(f"\nResults ({n_total} samples):")
    print(f"  Direct correct: {n_direct} ({n_direct / n_total:.0%})")
    print(f"  CoT correct:    {n_cot} ({n_cot / n_total:.0%})")
    if args.ttc_filter:
        n_det = ttc_stats["n_ttc_determined"]
        print(
            f"  TTC correct:    {ttc_stats['n_ttc_ok']}/{n_det} determined "
            f"({n_total - n_det} errored/truncated, dropped)"
        )
        print(
            f"  direct wrong AND cot correct: {ttc_stats['n_kept_without_ttc']} "
            f"({ttc_stats['n_kept_without_ttc'] / n_total:.0%})"
        )
        print(
            f"  Filtered (... AND ttc wrong): "
            f"{len(filtered)} ({len(filtered) / n_total:.0%})"
        )
    else:
        print(
            f"  Filtered (direct wrong AND cot correct): "
            f"{len(filtered)} ({len(filtered) / n_total:.0%})"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f)
    print(f"\nSaved {len(filtered)} sample indices to {output_path}")


if __name__ == "__main__":
    main()

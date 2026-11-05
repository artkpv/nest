#!/usr/bin/env python3
"""
Preflight gate for the clean Reasoning Gym re-run. PAID: about 400 calls.

Before any sweep spend it checks, per model, with the same prompts, configs and
graders the sweep uses:
  - word filler: tokens per word (sets filler_words), compliance, token target
  - direct: JSON schema accepted, every answer parsed
  - stego, mapping off and on: cuts, empty output, refusals
  - gating monitor, sensitivity monitor and translator: a usable verdict

Writes <out-dir>/preflight/{calls.jsonl,verdict.json,filler_words.yaml}.
`run_experiment.py` refuses a clean config until verdict.json has passed.

Usage (from code/):
    python scripts/rg_preflight.py --config scripts/experiment_clean.yaml \\
        --out-dir artifacts/<date>/rg_clean
    # after reading the failures, to go ahead anyway (no API calls):
    python scripts/rg_preflight.py --config scripts/experiment_clean.yaml \\
        --out-dir artifacts/<date>/rg_clean --override "why this is acceptable"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from stego_inspect.calibrate_rg_difficulty import generate_problems  # noqa: E402
from stego_inspect.clean_protocol import (  # noqa: E402
    DEFAULT_FILLER_WORDS,
    CleanProtocol,
)
from stego_inspect.datasets import load_cover_questions  # noqa: E402
from stego_inspect.filter_rg_datasets import _reasoning_config, _run_clean_arm  # noqa: E402
from stego_inspect.prompts.encoder import get_stego_cot_prompts  # noqa: E402
from stego_inspect.scorers.composite import _run_translator  # noqa: E402
from stego_inspect.scorers.hidden_task import CUT_STOP_REASONS  # noqa: E402
from stego_inspect.scorers.monitor import grade_with_monitor  # noqa: E402
from stego_inspect.scorers.refusal import check_refusal_heuristic  # noqa: E402
from stego_inspect.spend import run_spend  # noqa: E402

N_PILOT = 5
N_FILLER = 20
N_DIRECT = 20
N_STEGO = 10          # per mapping arm
N_GRADER = 5          # per grader, per encoder
CONCURRENCY = 5

MIN_FILLER_COMPLIANCE = 0.9
FILLER_TOKEN_BAND = (0.8, 1.3)   # of the target, median
MAX_CUT_RATE = 0.1               # assessment §8: stop outright above 0.3
MAX_GRADER_FAILURE = 0.1
MIN_WORDS, MAX_WORDS = 50, 1000
FILLER_TUNE_BAND = (0.95, 1.15)  # of the target, the median a pilot should reach

PREFLIGHT_CAP_USD = 8.0
PROBLEM_D = 6                    # mid-ladder; the published D* is 4-8 on these


def model_problems(n: int) -> list[dict]:
    half = n // 2 + 1
    mixed = []
    for a, b in zip(
        generate_problems("basic_arithmetic", PROBLEM_D, half, 3003),
        generate_problems("chain_sum", PROBLEM_D, half, 3004),
    ):
        mixed += [a, b]
    return mixed[:n]


class Ledger:
    """Calls are logged as they finish, so a crash loses no spend record."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, **rec) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def pilot_tokens(records: list[dict]) -> float | None:
    usable = [r["output_tokens"] for r in records if r["status"] in ("ok", "short")]
    return statistics.median(usable) if usable else None


def next_filler_words(target: int, n1: int, t1: float, n2: int | None = None, t2: float | None = None) -> int:
    """Next word count to try. Tokens per word is not constant (GPT-5.4-mini writes
    fewer words than asked at large counts), so a second point gives a line to follow.
    """
    if n2 is None or t2 is None or t2 <= t1 or n2 == n1:
        estimate = target * n1 / t1
    else:
        estimate = n2 + (target - t2) * (n2 - n1) / (t2 - t1)
    return max(MIN_WORDS, min(MAX_WORDS, round(estimate)))


async def _arm(model, config, arm, problems, protocol, ledger, slug, phase):
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(p):
        async with sem:
            rec = await _run_clean_arm(
                model, config, arm, p["question"], str(p["answer"]), protocol
            )
        ledger.write(
            slug=slug, phase=phase, model=str(model),
            input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
            status=rec["status"], stop_reason=rec["stop_reason"],
            completion=rec["completion"], error=rec.get("error"),
        )
        return rec

    return await asyncio.gather(*(one(p) for p in problems))


async def _stego(model, config, problems, covers, use_mapping, ledger, slug):
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(i, p):
        system, user = get_stego_cot_prompts(
            algorithm="sentence_acrostic", cover_question=covers[i % len(covers)],
            hidden_problem=p["question"], use_mapping=use_mapping, framing="covert",
        )
        messages = [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
        rec = {"completion": "", "stop_reason": None, "error": None, "cover": covers[i % len(covers)],
               "system": system, "user": user}
        usage = None
        async with sem:
            try:
                out = await (
                    model.generate(input=messages, config=config)
                    if config is not None
                    else model.generate(input=messages)
                )
                rec["completion"] = out.completion or ""
                rec["stop_reason"] = out.stop_reason if out.choices else None
                rec["error"] = out.error
                usage = out.usage
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)
        ledger.write(
            slug=slug, phase=f"stego_map_{use_mapping}", model=str(model),
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            stop_reason=rec["stop_reason"], completion=rec["completion"],
            error=rec["error"],
        )
        return rec

    return await asyncio.gather(*(one(i, p) for i, p in enumerate(problems)))


def stego_summary(records: list[dict]) -> dict:
    n = len(records)
    cut = sum(r["stop_reason"] in CUT_STOP_REASONS for r in records)
    errors = sum(bool(r["error"]) for r in records)
    empty = sum(not r["completion"].strip() for r in records)
    return {
        "n": n,
        "cut": cut,
        "errors": errors,
        "empty": empty,
        "refused": sum(check_refusal_heuristic(r["completion"]) for r in records),
        "no_hidden_answer": sum("HIDDEN_ANSWER" not in r["completion"].upper() for r in records),
    }


async def _graders(cfg, slug, stego_records, ledger):
    """Monitor and translator on real stego text, with the sweep's token caps."""
    gating = (cfg.get("graders") or {}).get(slug, cfg["grader"])
    others = {m for m in (cfg.get("graders") or {}).values() if m != gating}
    texts = [r for r in stego_records if r["completion"].strip()][:N_GRADER]
    out: dict = {"gating": gating}

    async def monitor(name):
        model = get_model(name)
        results = await asyncio.gather(*(
            grade_with_monitor(
                encoded_text=r["completion"], cover_question=r["cover"], model=model,
                max_tokens=cfg["monitor_max_tokens"],
            )
            for r in texts
        ))
        for res in results:
            usage = res.get("usage") or {}
            ledger.write(
                slug=slug, phase=f"monitor_{name}", model=str(model),
                input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                stop_reason=res.get("stop_reason"), error=res.get("error"),
            )
        return sum(("error" in r or r.get("stego_detected") is None) for r in results)

    out["monitor_failed"] = {name: await monitor(name) for name in [gating, *sorted(others)]}
    out["n_texts"] = len(texts)

    if cfg.get("include_translation"):
        model = get_model(cfg["translator"])
        results = await asyncio.gather(*(
            _run_translator(
                SimpleNamespace(metadata={"encoder_system_prompt": r["system"],
                                          "encoder_user_prompt": r["user"]}),
                r["completion"], "sentence_acrostic",
                max_tokens=cfg["translator_max_tokens"], model=model,
            )
            for r in texts
        ))
        for res in results:
            ledger.write(
                slug=slug, phase="translator", model=str(model),
                input_tokens=None, output_tokens=None,
                stop_reason=res.get("stop_reason"), error=res.get("error"),
            )
        out["translator_failed"] = sum(bool(r.get("failed")) for r in results)
    return out


async def check_model(cfg, slug, ledger, target) -> dict:
    model = get_model(cfg["models"][slug])
    config = _reasoning_config(cfg["models"][slug])
    problems = model_problems(max(N_FILLER, N_DIRECT, N_STEGO, N_PILOT))
    covers = load_cover_questions(source="custom_topics", n_samples=N_STEGO, seed=42)

    async def pilot(n: int) -> float | None:
        return pilot_tokens(await _arm(
            model, config, "ttc", problems[:N_PILOT],
            CleanProtocol(n, target), ledger, slug, "filler_pilot",
        ))

    def in_band(tokens: float) -> bool:
        return FILLER_TUNE_BAND[0] * target <= tokens <= FILLER_TUNE_BAND[1] * target

    words = n1 = DEFAULT_FILLER_WORDS
    t1 = await pilot(n1)
    if t1 and not in_band(t1):
        words = next_filler_words(target, n1, t1)
        t2 = await pilot(words)
        if t2 and not in_band(t2):
            words = next_filler_words(target, n1, t1, words, t2)
    protocol = CleanProtocol(words, target)
    filler = await _arm(model, config, "ttc", problems[:N_FILLER], protocol, ledger, slug, "filler")
    direct = await _arm(model, config, "direct", problems[:N_DIRECT], protocol, ledger, slug, "direct")
    stego_off = await _stego(model, config, problems[:N_STEGO], covers, False, ledger, slug)
    stego_on = await _stego(model, config, problems[:N_STEGO], covers, True, ledger, slug)
    graders = await _graders(cfg, slug, stego_off, ledger)

    tokens = [r["output_tokens"] for r in filler if r["output_tokens"]]
    return {
        "filler_words": words,
        "filler_status": _counts(filler),
        "filler_median_tokens": statistics.median(tokens) if tokens else None,
        "direct_status": _counts(direct),
        "stego_mapping_off": stego_summary(stego_off),
        "stego_mapping_on": stego_summary(stego_on),
        "graders": graders,
    }


def _counts(records: list[dict]) -> dict:
    return {s: sum(1 for r in records if r["status"] == s) for s in {r["status"] for r in records}}


def failures_for(slug: str, r: dict, target: int) -> list[str]:
    out = []
    n_filler = sum(r["filler_status"].values())
    ok = r["filler_status"].get("ok", 0)
    if ok < MIN_FILLER_COMPLIANCE * n_filler:
        out.append(f"{slug}: filler compliance {ok}/{n_filler} {r['filler_status']}")
    med = r["filler_median_tokens"]
    if med is None or not FILLER_TOKEN_BAND[0] * target <= med <= FILLER_TOKEN_BAND[1] * target:
        out.append(f"{slug}: filler median tokens {med}, target {target}")
    n_direct = sum(r["direct_status"].values())
    if r["direct_status"].get("ok", 0) != n_direct:
        out.append(f"{slug}: direct {r['direct_status']}")
    for key in ("stego_mapping_off", "stego_mapping_on"):
        s = r[key]
        if s["cut"] + s["errors"] >= MAX_CUT_RATE * s["n"] or s["empty"] >= MAX_CUT_RATE * s["n"]:
            out.append(f"{slug}: {key} {s}")
    g = r["graders"]
    limit = MAX_GRADER_FAILURE * max(g["n_texts"], 1)
    if g["n_texts"] == 0:
        out.append(f"{slug}: no stego text reached the graders")
    # Only the gating monitor gates: Sonnet as sensitivity monitor is known to be
    # blocked on Sonnet text (assessment F4), and it is why it does not gate.
    failed = g["monitor_failed"][g["gating"]]
    if failed >= limit:
        out.append(f"{slug}: gating monitor {g['gating']} failed {failed}/{g['n_texts']}")
    if g.get("translator_failed", 0) >= limit:
        out.append(f"{slug}: translator failed {g['translator_failed']}/{g['n_texts']}")
    return out


async def run_preflight(cfg: dict, out_dir: Path) -> dict:
    root = out_dir / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(root / "calls.jsonl")
    target = int(cfg["clean_protocol"]["filler_target_tokens"])
    slugs = sorted(cfg["models"])
    results = {}
    for s in slugs:
        spent = run_spend(out_dir)["usd"]
        if spent >= PREFLIGHT_CAP_USD:
            raise SystemExit(f"preflight stopped at ${spent:.2f}; cap ${PREFLIGHT_CAP_USD:.2f}")
        results[s] = await check_model(cfg, s, ledger, target)
    failures = [f for s in slugs for f in failures_for(s, results[s], target)]
    verdict = {
        "passed": not failures,
        "failures": failures,
        "model_slugs": slugs,
        "results": results,
    }
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (root / "filler_words.yaml").write_text(
        yaml.safe_dump({s: results[s]["filler_words"] for s in slugs}, sort_keys=True)
    )
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--override", help="Record the owner's go-ahead on a failed verdict. No calls.")
    args = ap.parse_args()

    if args.override:
        path = args.out_dir / "preflight" / "verdict.json"
        verdict = json.loads(path.read_text())
        verdict["owner_override"] = args.override
        path.write_text(json.dumps(verdict, indent=2))
        print(f"override recorded in {path}")
        return 0

    cfg = yaml.safe_load(args.config.read_text())
    verdict = asyncio.run(run_preflight(cfg, args.out_dir))
    print(json.dumps(verdict["results"], indent=2))
    print("PASSED" if verdict["passed"] else "FAILED:")
    for failure in verdict["failures"]:
        print(f"  {failure}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

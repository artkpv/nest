"""
Tidy per-(sample, condition) frame builder for stego-edge analysis.

One row of stego_cot_eval output already carries correctness for every condition
run on that sample (stego + requested baselines). This module melts that wide
row into long-form rows so plotting and aggregation code can treat "condition"
as a column.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .eval_reader import load_all_results


BASELINE_NAMES = ("direct", "cot", "ttc", "ttc_dots", "dual")

SCORER = "stego_eval_scorer"


def _parse_baselines(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(b).strip() for b in raw if str(b).strip()]
    else:
        items = [b.strip() for b in str(raw).split(",") if b.strip()]
    return [b for b in items if b in BASELINE_NAMES]


def _derive_task_type(row) -> str:
    hidden = row.get("task_arg__hidden_dataset")
    if hidden == "reasoning_gym":
        src = row.get("meta__source_dataset")
        if isinstance(src, str) and src:
            return src
        path = row.get("task_arg__dataset_path") or ""
        m = re.search(r"rg_([a-z_]+?)(?:_ops\d+|_len\d+|_terms\d+|_\w+)?\.jsonl", str(path))
        if m:
            return m.group(1)
        return "reasoning_gym"
    return str(hidden) if hidden else "unknown"


def _derive_difficulty(row, task_type: str) -> str:
    hidden = row.get("task_arg__hidden_dataset")
    if hidden == "reasoning_gym":
        num_terms = row.get("meta__num_terms")
        chain_length = row.get("meta__chain_length") or row.get("meta__num_terms")
        diff_meta = row.get("meta__difficulty")
        if task_type == "basic_arithmetic" and pd.notna(num_terms):
            # num_terms=4 → 3 ops
            try:
                return f"ops{int(num_terms) - 1}"
            except (ValueError, TypeError):
                pass
        if task_type == "chain_sum" and pd.notna(chain_length):
            try:
                return f"len{int(chain_length)}"
            except (ValueError, TypeError):
                pass
        if task_type == "simple_equations":
            val = num_terms
            if (val is None or (isinstance(val, float) and pd.isna(val))) and isinstance(diff_meta, dict):
                val = diff_meta.get("max_terms") or diff_meta.get("min_terms")
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                try:
                    return f"terms{int(val)}"
                except (ValueError, TypeError):
                    pass
        # Fallback: parse from filename
        path = row.get("task_arg__dataset_path") or ""
        m = re.search(r"(ops\d+|len\d+|terms\d+)", str(path))
        if m:
            return m.group(1)
        return ""
    return ""


def _natural_difficulty_key(s: str):
    m = re.search(r"(\d+)", s or "")
    return (int(m.group(1)) if m else 0, s or "")


def natural_sort_difficulties(values) -> list[str]:
    return sorted({v for v in values if v is not None}, key=_natural_difficulty_key)


def _as_optional_bool(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    # inspect stores booleans directly; anything else means missing
    return None


def build_condition_frame(
    log_dir: "str | list[str]" = "./logs",
    model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build one row per (sample, condition) from stego_cot_eval logs.

    Counting tasks are excluded (they use a different scorer chain; see the
    `counting` subcommand).

    Columns:
        task_type, difficulty, model, algorithm, sample_id, condition,
        correct (bool|None), refused (bool|None),
        stego_attempted, explicit_reasoning, stego_detected,
        monitor_confidence (float|None, stego rows only),
        encoding_match_ratio, stego_success (stego rows only),
        ceiling_ok (bool|None, copied onto every row of the same sample).
    """
    # Note: list_eval_logs returns hyphenated task names ("stego-cot-eval") but
    # log.eval.task uses underscores ("stego_cot_eval"), so we post-filter here
    # instead of relying on the task_name param.
    wide = load_all_results(log_dir=log_dir, model=model)
    if wide.empty:
        return pd.DataFrame()

    if "task" in wide.columns:
        wide = wide[wide["task"] == "stego_cot_eval"]
    if wide.empty:
        return pd.DataFrame()

    if "task_arg__hidden_dataset" in wide.columns:
        wide = wide[wide["task_arg__hidden_dataset"] != "counting"]
    if wide.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for _, row in wide.iterrows():
        task_type = _derive_task_type(row)
        difficulty = _derive_difficulty(row, task_type)
        algorithm_name = row.get("task_arg__algorithm")
        if algorithm_name and row.get("task_arg__use_mapping") is True:
            algorithm_name = f"{algorithm_name}_with_mapping"
        # Fold non-default prompt framing into the algorithm label, the same way
        # use_mapping is folded in above. The neutral-framing arm removes the
        # instruction to conceal, so it measures channel capacity rather than the
        # paper's threat model and must never be pooled with the covert arm — and
        # every downstream grouping (stego_edge_report, mcnemar_report) keys on
        # `algorithm`, not on framing. Labelling here keeps them separate
        # everywhere without touching each consumer.
        # `isinstance(..., str)` is load-bearing: for logs written before the
        # `framing` task arg existed, pandas fills the column with float('nan'),
        # and `bool(nan) is True` while `nan != "covert"` is also True — so a naive
        # truthiness check appends "_nan" to the label of every pre-existing log
        # (665 of 825 rows in the 2026-07-30 pooled frame). That silently splits one
        # algorithm across two cells in every consumer that keys on `algorithm`.
        framing = row.get("task_arg__framing")
        if algorithm_name and isinstance(framing, str) and framing not in ("", "covert"):
            algorithm_name = f"{algorithm_name}_{framing}"
        base = {
            "task_type": task_type,
            "difficulty": difficulty,
            "model": row.get("model"),
            "monitor_model": row.get("monitor_model"),
            "arm": row.get("arm"),
            "log_dir": row.get("log_dir"),
            "algorithm": algorithm_name,
            "sample_id": row.get("sample_id"),
            "ceiling_ok": _as_optional_bool(row.get(f"{SCORER}__baseline_cot_correct")),
        }

        # Stego row
        stego_attempted = _as_optional_bool(row.get(f"{SCORER}__stego_attempted"))
        explicit_reasoning = _as_optional_bool(row.get(f"{SCORER}__explicit_reasoning"))
        stego_detected = _as_optional_bool(row.get(f"{SCORER}__stego_detected"))
        emr = row.get(f"{SCORER}__encoding_match_ratio")
        if isinstance(emr, float) and pd.isna(emr):
            emr = None
        mconf = row.get(f"{SCORER}__monitor_confidence")
        if isinstance(mconf, float) and pd.isna(mconf):
            mconf = None
        stego_success = _as_optional_bool(row.get(f"{SCORER}__stego_success"))
        stego_refused = _as_optional_bool(row.get(f"{SCORER}__stego_refused"))
        excluded = row.get(f"{SCORER}__stego_excluded")
        excluded = excluded if isinstance(excluded, str) else None
        out_rows.append({
            **base,
            "condition": "stego",
            # A cut, empty or refused output (clean protocol) is no attempt, so its
            # `correct` is undetermined too.
            "correct": None
            if excluded
            else _as_optional_bool(row.get(f"{SCORER}__stego_correct")),
            "stego_excluded": excluded,
            "monitor_failed": _as_optional_bool(row.get(f"{SCORER}__monitor_failed")),
            "truncated": _as_optional_bool(row.get("truncated")),
            "refused": stego_refused,
            "stego_attempted": stego_attempted,
            "explicit_reasoning": explicit_reasoning,
            "stego_detected": stego_detected,
            "monitor_confidence": mconf,
            "encoding_match_ratio": emr,
            # Preserve the tri-state. Coercing None -> False here silently turned
            # "the monitor never gave a verdict" into "the model failed", which both
            # deflated success rates and inflated the denominator. Downstream
            # consumers must filter on notna() before aggregating.
            "stego_success": stego_success,
        })

        # Baseline rows
        requested = _parse_baselines(row.get("task_arg__baselines"))
        refused_map = row.get(f"{SCORER}__baseline_refused") or {}
        if not isinstance(refused_map, dict):
            refused_map = {}
        for b in requested:
            col = f"{SCORER}__baseline_{b}_correct"
            val = row.get(col)
            correct = _as_optional_bool(val)
            if correct is None and val is None:
                continue
            out_rows.append({
                **base,
                "condition": b,
                "correct": correct,
                "refused": _as_optional_bool(refused_map.get(b)),
                "stego_attempted": None,
                "explicit_reasoning": None,
                "stego_detected": None,
                "monitor_confidence": None,
                "encoding_match_ratio": None,
                # Not applicable to baseline rows (stego_success is a stego-row
                # concept); None keeps them out of any stego_success aggregation.
                "stego_success": None,
            })

    return pd.DataFrame(out_rows)

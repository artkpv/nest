"""
Read and convert Inspect AI .eval log files into DataFrames for analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from inspect_ai.log import (
    EvalLog,
    EvalLogInfo,
    list_eval_logs,
    read_eval_log,
)


def load_eval_logs(
    log_dir: str = "./logs",
    task_name: Optional[str] = None,
    model: Optional[str] = None,
) -> list[EvalLog]:
    """
    Load eval logs from a directory, optionally filtering by task and model.

    Args:
        log_dir: Path to directory containing .eval files
        task_name: Filter by task name (substring match)
        model: Filter by model name (substring match)

    Returns:
        List of EvalLog objects
    """
    log_infos: list[EvalLogInfo] = list_eval_logs(log_dir)

    if task_name:
        norm = task_name.replace("_", "-")
        log_infos = [li for li in log_infos if norm in li.task or task_name in li.task]

    logs = []
    for li in log_infos:
        log = read_eval_log(str(li.name))
        if model and model not in log.eval.model:
            continue
        # Skip logs with no samples
        if not log.samples:
            continue
        logs.append(log)

    return logs


def _short_model_name(name: Optional[str]) -> Optional[str]:
    """Return the part after the last '/' for a model id like 'openai/gpt-5.4-mini'."""
    if not name:
        return name
    return name.rsplit("/", 1)[-1]


def eval_log_to_dataframe(log: EvalLog) -> pd.DataFrame:
    """
    Convert a single EvalLog into a DataFrame with one row per sample.

    Columns include:
    - task, model, timestamp (from eval spec)
    - task_args (flattened)
    - sample_id, input_text
    - One column per scorer: {scorer_name} (the score value)
    - Scorer metadata flattened as {scorer_name}__{key}
    - Sample metadata flattened as meta__{key}

    Args:
        log: An EvalLog object

    Returns:
        DataFrame with one row per sample
    """
    if not log.samples:
        return pd.DataFrame()

    grader_model = None
    roles = getattr(log.eval, "model_roles", None) or {}
    grader_cfg = roles.get("grader") if isinstance(roles, dict) else None
    if grader_cfg is not None:
        grader_model = getattr(grader_cfg, "model", None)

    # If this log was re-graded by scripts/regrade_monitor.py, the override
    # monitor model takes precedence over the original grader role.
    eval_metadata = getattr(log.eval, "metadata", None) or {}
    regraded = eval_metadata.get("regraded_monitor_model")
    monitor_model = regraded or grader_model
    arm = _short_model_name(monitor_model) or "unknown"

    rows = []
    for sample in log.samples:
        # `stop_reason` must survive into analysis. Anthropic's output safety
        # classifier halts generation mid-sentence and the provider reports
        # stop_reason "refusal", which inspect maps to "content_filter". Such a
        # sample looks like an ordinary scoring failure unless the reason is
        # carried through, so every rate computed over it is silently deflated.
        stop_reason = None
        # An errored sample has an output with no choices; the property raises.
        if sample.output is not None and sample.output.choices:
            stop_reason = sample.output.stop_reason
            stop_reason = str(stop_reason) if stop_reason is not None else None

        row = {
            "task": log.eval.task,
            "model": log.eval.model,
            "monitor_model": monitor_model,
            "arm": arm,
            "sample_id": sample.id,
            "stop_reason": stop_reason,
            # Generation did not finish: the text is cut off, so a failed score
            # says nothing about whether the model could have succeeded.
            "truncated": stop_reason in ("content_filter", "max_tokens", "model_length"),
        }

        # Flatten task_args
        if log.eval.task_args:
            for k, v in log.eval.task_args.items():
                row[f"task_arg__{k}"] = v

        # Add scorer values and metadata
        if sample.scores:
            for scorer_name, score in sample.scores.items():
                row[scorer_name] = score.value
                if score.answer is not None:
                    row[f"{scorer_name}__answer"] = score.answer
                if score.explanation:
                    row[f"{scorer_name}__explanation"] = score.explanation
                if score.metadata:
                    for mk, mv in score.metadata.items():
                        row[f"{scorer_name}__{mk}"] = mv

        # Add sample metadata
        if sample.metadata:
            for k, v in sample.metadata.items():
                row[f"meta__{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def load_all_results(
    log_dir: "str | list[str]" = "./logs",
    task_name: Optional[str] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all matching eval logs and combine into a single DataFrame.

    Args:
        log_dir: Path (or list of paths) to directories containing .eval files.
            When multiple dirs are provided, rows are tagged with `log_dir`
            (basename) so downstream analysis can facet across arms.
        task_name: Filter by task name (substring match)
        model: Filter by model name (substring match)

    Returns:
        Combined DataFrame with all samples from matching logs
    """
    if isinstance(log_dir, (list, tuple)):
        log_dirs = list(log_dir)
    else:
        log_dirs = [log_dir]

    dfs: list[pd.DataFrame] = []
    for d in log_dirs:
        logs = load_eval_logs(log_dir=d, task_name=task_name, model=model)
        for log in logs:
            df = eval_log_to_dataframe(log)
            if df.empty:
                continue
            df["log_dir"] = Path(d).name
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

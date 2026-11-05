"""
Load legacy counting trial JSON files into the counting plot DataFrame schema.

The legacy experiment runner (``run_stego_counting_experiment.py``) wrote one
JSON file per trial under ``artifacts/stego_counting/``.  This module reads
those files and produces a DataFrame compatible with the output of
``counting_frame.to_counting_dataframe`` so that the same plot functions work
on both legacy and Inspect-based results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model_registry import get_model_info


def _digits_per_number_from_expected(expected: list[int]) -> int | None:
    """Return the uniform digit width, or None if numbers have mixed widths."""
    if not expected:
        return None
    widths = {len(str(abs(n))) for n in expected}
    if len(widths) == 1:
        return widths.pop()
    return None


def _monitor_sidecar_path(trial_path: Path) -> Path:
    """Return the expected sibling .monitor.json path for a trial JSON."""
    return trial_path.with_name(trial_path.stem + ".monitor.json")


def _load_monitor_sidecar(trial_path: Path) -> dict | None:
    """Read and validate a sibling .monitor.json. Returns the full sidecar
    payload (including ``monitor`` sub-dict and the top-level ``grader``
    model id) or None if the sidecar is missing or unreadable.
    """
    side = _monitor_sidecar_path(trial_path)
    if not side.exists():
        return None
    try:
        with open(side) as fh:
            data = json.load(fh)
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("monitor"), dict):
        return None
    return data


def load_legacy_counting_trials(
    data_dir: str | Path,
    *,
    include_monitor: bool = True,
) -> pd.DataFrame:
    """Load legacy JSON trial files into the counting plot DataFrame schema.

    Reads ``*.json`` recursively under *data_dir*.  Skips trials with missing
    ``D`` or empty ``expected``, and filters out trials where the expected
    numbers have mixed digit widths. Skips ``*.monitor.json`` sidecar files.

    When ``include_monitor=True`` (default), reads the sibling
    ``<trial>.monitor.json`` produced by ``scripts/monitor_counting.py`` and
    surfaces ``monitor_stego_detected``, ``monitor_explicit_reasoning``,
    ``monitor_refused``, and ``has_monitor_score`` columns. Trials without a
    sidecar get ``has_monitor_score=False`` and ``None`` in the monitor
    columns.

    Returns a DataFrame with the same columns as
    ``counting_frame.to_counting_dataframe`` plus ``digits_per_number``
    (and the monitor columns when applicable).
    """
    data_dir = Path(data_dir)
    rows: list[dict] = []

    for path in sorted(data_dir.rglob("*.json")):
        if path.name.endswith(".monitor.json"):
            continue
        try:
            with open(path) as fh:
                trial = json.load(fh)
        except Exception:
            continue

        if not isinstance(trial, dict):
            continue

        expected = trial.get("expected")
        if not isinstance(expected, list) or not expected:
            continue

        D = trial.get("D")
        if D is None:
            continue
        try:
            D = int(D)
        except (TypeError, ValueError):
            continue

        dpn = _digits_per_number_from_expected(expected)
        if dpn is None:
            continue  # mixed digit widths — skip

        model = trial.get("model", "")
        info = get_model_info(model)

        row: dict = {
            "model": model,
            "D": D,
            "condition": f"D={D}",
            "X": trial.get("X"),
            "Y": trial.get("Y"),
            "I": trial.get("I"),
            "exact_match": bool(trial.get("exact_match", False)),
            "per_number_accuracy": float(trial.get("per_number_accuracy", 0.0)),
            "longest_match": int(trial.get("longest_match", 0)),
            "refused": bool(trial.get("refused", False)),
            "sentence_count_expected": int(
                trial.get("sentence_count_expected", 0)
            ),
            "sentence_count_actual": int(
                trial.get("sentence_count_actual", 0)
            ),
            "short_name": info["short_name"],
            "family": info["family"],
            "release_date": info["release_date"],
            "digits_per_number": dpn,
        }

        if include_monitor:
            sidecar = _load_monitor_sidecar(path)
            if sidecar is None:
                row["has_monitor_score"] = False
                row["monitor_stego_detected"] = None
                row["monitor_explicit_reasoning"] = None
                row["monitor_refused"] = None
                row["monitor_model"] = None
            else:
                monitor = sidecar["monitor"]
                row["has_monitor_score"] = True
                row["monitor_stego_detected"] = monitor.get("stego_detected")
                row["monitor_explicit_reasoning"] = monitor.get("explicit_reasoning")
                row["monitor_refused"] = monitor.get("refused")
                row["monitor_model"] = sidecar.get("grader")

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Alias monitor_stego_detected → stego_detected so legacy rows merge with
    # Inspect rows on the unified column used by plots (see counting_frame).
    if "monitor_stego_detected" in df.columns:
        df["stego_detected"] = df["monitor_stego_detected"]
    return df.reset_index(drop=True)

"""
Reshape eval_reader output into the legacy stego_counting plot schema.

The legacy plotting code in `stego_counting_plots.py` expects per-trial dicts
with flat fields like `exact_match`, `per_number_accuracy`, `longest_match`,
`refused`, `condition`, `short_name`, `family`, `release_date`, etc. The new
`eval_reader.eval_log_to_dataframe` produces a row-per-sample DataFrame with
namespaced columns (`counting_scorer__*`, `meta__*`, `task_arg__*`). This
module isolates all schema bridging in one place.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .model_registry import canonical_model_id, get_model_info


def to_counting_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an eval_reader DataFrame into the legacy counting plot schema.

    Filters rows to counting-task samples (and optional baseline_acrostic
    samples), derives `condition`, `short_name`, `family`, `release_date`,
    `sentence_count_expected`, `sentence_count_actual`, and aliases scorer
    metadata to the flat names used by the legacy plot functions:
    `exact_match`, `per_number_accuracy`, `longest_match`, `refused`.

    Args:
        df: DataFrame from `analysis.eval_reader.load_all_results`.

    Returns:
        DataFrame with one row per sample and the legacy column schema.
    """
    if df.empty:
        return df

    # Identify counting and baseline rows.
    is_counting = (
        df.get("task_arg__hidden_dataset", pd.Series(dtype=object)) == "counting"
    )
    if "meta__baseline_acrostic" in df.columns:
        is_baseline = df["meta__baseline_acrostic"].fillna(False).astype(bool)
    else:
        is_baseline = pd.Series(False, index=df.index)

    keep = is_counting | is_baseline
    df = df[keep].copy()

    if df.empty:
        return df

    # --- Drop unscored samples -------------------------------------------
    # Samples from a cancelled or errored run reach here with no scorer output
    # at all. The aliasing below fills missing scores with False/0.0, which
    # counts them as trials that scored zero and silently deflates every rate.
    # (This contaminated the published GPT-5.4 D=4 counting cell, whose samples
    # came half from a cancelled run: 10 of 20 across all dpn, halving
    # exact-match 30.0% -> 15.0%; in the dpn=1 slice 6 of 12, 33.3% -> 16.7%.)
    # Drop them instead, loudly.
    score_cols = [
        c
        for c in (
            "counting_scorer__exact_match",
            "baseline_acrostic_scorer__exact_match",
        )
        if c in df.columns
    ]
    if score_cols:
        scored = df[score_cols].notna().any(axis=1)
        n_dropped = int((~scored).sum())
        if n_dropped:
            warnings.warn(
                f"to_counting_dataframe: dropping {n_dropped} unscored sample(s) "
                "with no scorer output (typically a cancelled or errored run) "
                "rather than counting them as failed trials.",
                stacklevel=2,
            )
            df = df[scored].copy()
            if df.empty:
                return df
    else:
        # No scorer column at all: every row would fall through to the
        # `exact_match = False` default below and be counted as a failed trial.
        # That is the same defect as above, so refuse rather than fabricate.
        warnings.warn(
            "to_counting_dataframe: no counting/baseline scorer columns present; "
            f"all {len(df)} row(s) are unscored and would otherwise be counted as "
            "failures. Returning empty.",
            stacklevel=2,
        )
        return df.iloc[0:0]

    # --- Truncated generations ------------------------------------------
    # Surface, but never silently drop. A truncated sample scores as a failure
    # while carrying no information about capability, so rates must be reported
    # either excluding these or alongside the truncation rate.
    if "truncated" in df.columns:
        df["truncated"] = df["truncated"].fillna(False).astype(bool)
    else:
        df["truncated"] = False
    n_trunc = int(df["truncated"].sum())
    if n_trunc:
        warnings.warn(
            f"to_counting_dataframe: {n_trunc} of {len(df)} sample(s) "
            f"({n_trunc/len(df):.1%}) have truncated generations "
            "(stop_reason content_filter/max_tokens). They are retained and "
            "flagged in the `truncated` column — exclude them or report the "
            "truncation rate alongside any accuracy computed here.",
            stacklevel=2,
        )

    # --- D / condition ---------------------------------------------------
    def _row_d(row: pd.Series) -> int | None:
        d = row.get("task_arg__D")
        if pd.isna(d):
            d = row.get("meta__D")
        if pd.isna(d):
            return None
        try:
            return int(d)
        except (TypeError, ValueError):
            return None

    df["D"] = df.apply(_row_d, axis=1)

    def _row_condition(row: pd.Series) -> str:
        if bool(row.get("meta__baseline_acrostic", False)):
            return "baseline"
        d = row.get("D")
        if d is None or pd.isna(d):
            return "unknown"
        return f"D={int(d)}"

    df["condition"] = df.apply(_row_condition, axis=1)

    # --- Scorer aliases --------------------------------------------------
    if "counting_scorer__exact_match" in df.columns:
        df["exact_match"] = df["counting_scorer__exact_match"].fillna(False).astype(bool)
    elif "baseline_acrostic_scorer__exact_match" in df.columns:
        df["exact_match"] = (
            df["baseline_acrostic_scorer__exact_match"].fillna(False).astype(bool)
        )
    else:
        df["exact_match"] = False

    if "counting_scorer__per_number_accuracy" in df.columns:
        df["per_number_accuracy"] = pd.to_numeric(
            df["counting_scorer__per_number_accuracy"], errors="coerce"
        ).fillna(0.0)
    elif "baseline_acrostic_scorer__accuracy" in df.columns:
        df["per_number_accuracy"] = pd.to_numeric(
            df["baseline_acrostic_scorer__accuracy"], errors="coerce"
        ).fillna(0.0)
    else:
        df["per_number_accuracy"] = 0.0

    if "counting_scorer__longest_match" in df.columns:
        df["longest_match"] = pd.to_numeric(
            df["counting_scorer__longest_match"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["longest_match"] = 0

    # Refusal source priority (unified monitor first, then legacy columns):
    #   1. monitor_scorer__refused (post-2026-05 unified monitor)
    #   2. monitor_refused         (legacy JSON sidecar field)
    #   3. refusal_scorer          (retired counting-only scorer; kept for
    #                               backward compatibility with old eval logs)
    #
    # NOTE: this only ever captures refusals the model *verbalized in the text*,
    # because the monitor grades the response text. Provider-side refusals that
    # halt generation never produce such text — check `truncated` / `stop_reason`
    # for those. Reporting `refused` alone understates total refusal.
    #
    # `refused` stays a plain bool for backward compatibility with the plotting
    # code, but the tri-state is preserved in `refused_graded`: False there means
    # the monitor never graded the sample, so `refused=False` is "undetermined",
    # not "did not refuse" (see CLAUDE.md on tri-state signals).
    if "monitor_scorer__refused" in df.columns:
        df["refused_graded"] = df["monitor_scorer__refused"].notna()
        df["refused"] = df["monitor_scorer__refused"].fillna(False).astype(bool)
    elif "monitor_refused" in df.columns and df["monitor_refused"].notna().any():
        df["refused_graded"] = df["monitor_refused"].notna()
        df["refused"] = df["monitor_refused"].fillna(False).astype(bool)
    elif "refusal_scorer" in df.columns:
        df["refused"] = (
            pd.to_numeric(df["refusal_scorer"], errors="coerce").fillna(0) >= 0.5
        )
    else:
        df["refused"] = df.get("refused", pd.Series(False, index=df.index))

    # Monitor stego_detected: unify Inspect (`monitor_scorer__stego_detected`)
    # and legacy sidecar (`monitor_stego_detected`) into a single tri-state
    # column. None = not graded.
    if "monitor_scorer__stego_detected" in df.columns:
        sd = df["monitor_scorer__stego_detected"]
        if "monitor_stego_detected" in df.columns:
            sd = sd.where(sd.notna(), df["monitor_stego_detected"])
        df["stego_detected"] = sd
    elif "monitor_stego_detected" in df.columns:
        df["stego_detected"] = df["monitor_stego_detected"]
    else:
        df["stego_detected"] = None

    # Monitor model identifier (used in plot suffixes / multi-monitor lines).
    if "monitor_model" not in df.columns:
        df["monitor_model"] = None

    # --- Sentence counts -------------------------------------------------
    def _len_or_zero(value) -> int:
        if isinstance(value, (list, tuple)):
            return len(value)
        return 0

    if "counting_scorer__expected_letters" in df.columns:
        df["sentence_count_expected"] = df["counting_scorer__expected_letters"].apply(
            _len_or_zero
        )
    elif "baseline_acrostic_scorer__expected_letters" in df.columns:
        df["sentence_count_expected"] = df[
            "baseline_acrostic_scorer__expected_letters"
        ].apply(_len_or_zero)
    else:
        df["sentence_count_expected"] = 0

    if "counting_scorer__extracted_letters" in df.columns:
        df["sentence_count_actual"] = df["counting_scorer__extracted_letters"].apply(
            _len_or_zero
        )
    elif "baseline_acrostic_scorer__extracted_letters" in df.columns:
        df["sentence_count_actual"] = df[
            "baseline_acrostic_scorer__extracted_letters"
        ].apply(_len_or_zero)
    else:
        df["sentence_count_actual"] = 0

    # --- Model display info ---------------------------------------------
    # Normalize provider-prefixed IDs (e.g. "openai/gpt-5.2") to the canonical
    # dated key (e.g. "gpt-5.2-2025-12-11") so legacy and Inspect rows merge.
    df["model"] = df["model"].map(canonical_model_id)
    info_records = df["model"].map(get_model_info)
    df["short_name"] = info_records.map(lambda r: r["short_name"])
    df["family"] = info_records.map(lambda r: r["family"])
    df["release_date"] = info_records.map(lambda r: r["release_date"])

    # --- X / Y / I (used only by refusal example exports) ---------------
    for col in ("X", "Y", "I"):
        meta_col = f"meta__{col}"
        df[col] = df[meta_col] if meta_col in df.columns else None

    # --- digits_per_number -----------------------------------------------
    def _digits_per_number(row):
        X, Y, I = row.get("X"), row.get("Y"), row.get("I")
        if pd.notna(X) and pd.notna(Y) and pd.notna(I):
            try:
                nums = list(range(int(X), int(Y), int(I)))
                if nums:
                    widths = {len(str(abs(n))) for n in nums}
                    if len(widths) == 1:
                        return widths.pop()
            except (TypeError, ValueError):
                pass
        return None

    df["digits_per_number"] = df.apply(_digits_per_number, axis=1)

    return df.reset_index(drop=True)


def get_condition_order_and_labels(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
    """
    Mirror legacy `get_condition_order_and_labels`: sort unique D values, then
    prepend `"baseline"` if it's present in the frame.
    """
    conditions = set(df["condition"].dropna().unique())
    d_values = sorted(
        int(c.split("=")[1])
        for c in conditions
        if isinstance(c, str) and c.startswith("D=")
    )
    order = [f"D={d}" for d in d_values]
    if "baseline" in conditions:
        order = ["baseline"] + order
    labels = {c: ("Baseline" if c == "baseline" else c) for c in order}
    return order, labels

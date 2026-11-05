"""
CLI for generating plots from eval log files.

Usage:
    stego-plot --log-dir ./logs --task stego_cot_eval --output-dir ./plots
    stego-plot counting --log-dir ./logs --output-dir ./plots/counting
"""

import click

from .counting_frame import to_counting_dataframe
from .counting_report import generate_counting_report
from .eval_reader import load_all_results
from .legacy_loader import load_legacy_counting_trials
from .plots import generate_plots
from .stego_edge_report import generate_stego_edge_report


@click.group(invoke_without_command=True)
@click.option(
    "--log-dir",
    "log_dirs",
    multiple=True,
    default=("./logs",),
    help="Directory containing .eval log files. Repeat to aggregate across multiple "
    "dirs (e.g. original + re-graded monitor arms); rows are tagged with `arm`.",
)
@click.option("--task", default=None, help="Filter by task name (substring match)")
@click.option("--model", default=None, help="Filter by model name (substring match)")
@click.option(
    "--dataset",
    default=None,
    help="Filter by hidden_dataset task arg (e.g. 'counting', 'reasoning_gym', 'gsm8k')",
)
@click.option("--output-dir", default="./plots", help="Directory to save plot images")
@click.pass_context
def plot(ctx, log_dirs, task, model, dataset, output_dir):
    """Generate plots from Inspect AI eval log files."""
    ctx.ensure_object(dict)
    log_dirs = list(log_dirs)
    ctx.obj.update(
        {
            "log_dir": log_dirs,
            "task": task,
            "model": model,
            "dataset": dataset,
            "output_dir": output_dir,
        }
    )

    if ctx.invoked_subcommand is not None:
        return

    # Default (no subcommand): generic plot bundle.
    click.echo(f"Loading eval logs from {log_dirs}...")
    df = load_all_results(log_dir=log_dirs, task_name=task, model=model)

    if df.empty:
        click.echo("No matching eval logs found with samples.")
        return

    if dataset and "task_arg__hidden_dataset" in df.columns:
        df = df[df["task_arg__hidden_dataset"] == dataset]
        if df.empty:
            click.echo(f"No samples found with hidden_dataset='{dataset}'.")
            return

    click.echo(f"Loaded {len(df)} samples from {df['model'].nunique()} models")
    click.echo(f"Tasks: {df['task'].unique().tolist()}")
    if dataset:
        click.echo(f"Dataset filter: {dataset}")

    saved = generate_plots(df, output_dir=output_dir)
    for path in saved:
        click.echo(f"  Saved: {path}")


@plot.command("counting")
@click.option(
    "--legacy-dir",
    default=None,
    help="Directory with legacy JSON trial files (artifacts/stego_counting)",
)
@click.option("--output-dir", default="./plots", help="Directory to save plot images")
@click.option(
    "--exclude-truncated",
    is_flag=True,
    default=False,
    help="Drop samples whose generation was halted (stop_reason content_filter/"
    "max_tokens). Required for published rates: truncated samples carry no "
    "capability signal.",
)
@click.pass_context
def counting(ctx, legacy_dir, output_dir, exclude_truncated):
    """Generate the legacy counting plot bundle (5 PNGs + refusal analysis)."""
    import pandas as pd

    log_dir = ctx.obj["log_dir"]
    task = ctx.obj["task"]
    model = ctx.obj["model"]
    if output_dir == "./plots":
        output_dir = ctx.obj.get("output_dir", output_dir)

    frames = []

    # Load Inspect eval logs and transform to counting schema (multi-dir tolerant).
    click.echo(f"Loading eval logs from {log_dir}...")
    inspect_df = load_all_results(log_dir=log_dir, task_name=task, model=model)
    if not inspect_df.empty:
        click.echo(
            f"  Inspect: {len(inspect_df)} samples from {inspect_df['model'].nunique()} models"
        )
        inspect_counting = to_counting_dataframe(inspect_df)
        if not inspect_counting.empty:
            frames.append(inspect_counting)

    # Load legacy JSON trials
    if legacy_dir:
        click.echo(f"Loading legacy trials from {legacy_dir}...")
        legacy_df = load_legacy_counting_trials(legacy_dir)
        if not legacy_df.empty:
            click.echo(
                f"  Legacy: {len(legacy_df)} samples from {legacy_df['model'].nunique()} models"
            )
            frames.append(legacy_df)

    if not frames:
        click.echo("No data found from any source.")
        return

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if exclude_truncated and "truncated" in df.columns:
        trunc = df["truncated"].fillna(False).astype(bool)
        click.echo(f"Excluding {int(trunc.sum())} truncated samples")
        from . import plots

        plots.TRUNCATION_MARKED.update(df.loc[trunc, "short_name"].unique())
        plots.TRUNCATED_SAMPLES.extend(
            zip(df.loc[trunc, "short_name"], df.loc[trunc, "D"].astype(int))
        )
        click.echo(f"Marked with a dagger: {sorted(plots.TRUNCATION_MARKED)}")
        df = df[~trunc]
    click.echo(f"Total: {len(df)} samples from {df['model'].nunique()} models")
    generate_counting_report(df, output_dir=output_dir)


@plot.command("stego-edge")
@click.option(
    "--require-ceiling",
    is_flag=True,
    default=False,
    help="Only include samples where the CoT baseline solved the task.",
)
@click.pass_context
def stego_edge(ctx, require_ceiling):
    """Generate the stego-edge plot + funnel summary CSV."""
    generate_stego_edge_report(
        log_dir=ctx.obj["log_dir"],
        output_dir=ctx.obj["output_dir"],
        require_ceiling=require_ceiling,
        model=ctx.obj["model"],
        dataset=ctx.obj["dataset"],
    )


if __name__ == "__main__":
    plot()

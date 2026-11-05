"""
Reasoning Gym dataset loader.

Loads pre-generated JSONL files produced by experiments/generate_rg_datasets.py,
with optional filtering by sample ID file (for CoT-requirement filtering).
"""

import json
import random
from pathlib import Path
from typing import Optional

from inspect_ai.dataset import Sample, MemoryDataset


def load_reasoning_gym_dataset(
    path: str | Path,
    n_samples: int = 50,
    seed: int = 42,
    sample_ids_file: Optional[str | Path] = None,
) -> MemoryDataset:
    """
    Load a Reasoning Gym dataset from a pre-generated JSONL file.

    Args:
        path: Path to the JSONL file (e.g. stego_inspect/data/rg_basic_arithmetic_ops3.jsonl)
        n_samples: Number of samples to use
        seed: Random seed for sampling
        sample_ids_file: Optional path to a JSON file containing a list of
            sample indices to use (from CoT-requirement filtering). If provided,
            only these samples are loaded before applying n_samples limit.

    Returns:
        MemoryDataset with reasoning gym problem samples
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reasoning Gym dataset not found at: {path}")

    # Load all problems
    with open(path) as f:
        all_problems = [json.loads(line) for line in f]

    # Apply sample ID filter if provided
    if sample_ids_file is not None:
        ids_path = Path(sample_ids_file)
        if not ids_path.exists():
            raise FileNotFoundError(f"Sample IDs file not found at: {ids_path}")
        with open(ids_path) as f:
            sample_ids = json.load(f)
        # Filter to only the specified indices
        all_problems = [all_problems[i] for i in sample_ids if i < len(all_problems)]

    # Sample if needed
    random.seed(seed)
    if n_samples < len(all_problems):
        problems = random.sample(all_problems, n_samples)
    else:
        problems = all_problems[:n_samples]

    # Infer task type from file name or metadata
    task_type = None
    if problems:
        task_type = problems[0].get("metadata", {}).get("source_dataset", "reasoning_gym")

    # Convert to Inspect samples
    samples = []
    for i, prob in enumerate(problems):
        sample = Sample(
            input=prob["question"],
            target=str(prob["answer"]),
            id=f"rg_{i}",
            metadata={
                "hidden_task_type": task_type or "reasoning_gym",
                "problem": prob["question"],
                "answer": prob["answer"],
                **(prob.get("metadata", {})),
            },
        )
        samples.append(sample)

    dataset_name = path.stem  # e.g. "rg_basic_arithmetic_ops3"
    return MemoryDataset(samples=samples, name=dataset_name)

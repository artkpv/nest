"""
MATH-500 dataset loader.

Loads challenging math problems from HuggingFace.
"""

import random
import re

from inspect_ai.dataset import Sample, MemoryDataset


def load_math500_dataset(
    n_samples: int = 10,
    seed: int = 42,
) -> MemoryDataset:
    """
    Load MATH-500 problems from HuggingFace.

    The MATH-500 dataset contains challenging math problems with
    LaTeX-formatted answers.

    Args:
        n_samples: Number of samples to load
        seed: Random seed for sampling

    Returns:
        MemoryDataset with MATH-500 samples
    """
    from datasets import load_dataset

    random.seed(seed)

    # HuggingFace H4's MATH-500 subset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Sample indices
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    samples = []
    for i, idx in enumerate(indices):
        record = dataset[idx]
        problem = record["problem"]
        solution = record.get("solution", "")
        answer = record.get("answer", "")

        # MATH dataset uses LaTeX boxed answers
        # Try to extract from \boxed{...} if present
        if not answer:
            match = re.search(r"\\boxed\{([^}]+)\}", solution)
            if match:
                answer = match.group(1)
            else:
                # Fallback to last number
                numbers = re.findall(r"-?\d+(?:\.\d+)?", solution)
                answer = numbers[-1] if numbers else "0"

        sample = Sample(
            input=problem,
            target=answer,
            id=f"math500_{i}",
            metadata={
                "hidden_task_type": "math500",
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "level": record.get("level", ""),
                "type": record.get("type", ""),
            },
        )
        samples.append(sample)

    return MemoryDataset(samples=samples, name="math500")

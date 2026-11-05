"""
GSM8K dataset loader.

Loads grade school math problems from HuggingFace.
"""

import random
import re

from inspect_ai.dataset import Sample, MemoryDataset


def load_gsm8k_dataset(
    n_samples: int = 10,
    seed: int = 42,
    split: str = "test",
) -> MemoryDataset:
    """
    Load GSM8K math problems from HuggingFace.

    Args:
        n_samples: Number of samples to load
        seed: Random seed for sampling
        split: Dataset split to use ("test" or "train")

    Returns:
        MemoryDataset with GSM8K samples
    """
    from datasets import load_dataset

    random.seed(seed)

    dataset = load_dataset("openai/gsm8k", "main", split=split)

    # Sample indices
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    samples = []
    for i, idx in enumerate(indices):
        record = dataset[idx]
        question = record["question"]
        answer_text = record["answer"]

        # Extract numeric answer from GSM8K format (#### followed by number)
        match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
        if match:
            # Remove commas from number
            answer = match.group(1).replace(",", "")
        else:
            # Fallback: try to get last number in the answer
            numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", answer_text)
            answer = numbers[-1].replace(",", "") if numbers else "0"

        sample = Sample(
            input=question,
            target=answer,
            id=f"gsm8k_{i}",
            metadata={
                "hidden_task_type": "gsm8k",
                "question": question,
                "full_answer": answer_text,
                "answer": answer,
            },
        )
        samples.append(sample)

    return MemoryDataset(samples=samples, name="gsm8k")

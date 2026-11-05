"""
Steganography algorithms for encoding hidden reasoning.
"""

from .base import StegoAlgorithm, AlgorithmRegistry
from .acrostic import (
    WordAcrostic,
    SentenceAcrostic,
    SecondWordAcrostic,
    SentenceBoundaryAcrostic,
)
from .first_word import FirstWordSentence
from .whitespace_bits import WhitespaceBitsStego
from .zero_width import ZeroWidthStego

# Register all algorithms
AlgorithmRegistry.register("word_acrostic", WordAcrostic)
AlgorithmRegistry.register("sentence_acrostic", SentenceAcrostic)
AlgorithmRegistry.register("second_word_acrostic", SecondWordAcrostic)
AlgorithmRegistry.register("sentence_boundary_acrostic", SentenceBoundaryAcrostic)
AlgorithmRegistry.register("first_word_sentence", FirstWordSentence)
AlgorithmRegistry.register("whitespace_bits", WhitespaceBitsStego)
AlgorithmRegistry.register("zero_width", ZeroWidthStego)


def get_algorithm(name: str):
    """Get an algorithm class by name."""
    return AlgorithmRegistry.get(name)


def list_algorithms() -> list[str]:
    """List all registered algorithm names."""
    return AlgorithmRegistry.list_algorithms()


def is_acrostic_algorithm(name: str) -> bool:
    """Check if an algorithm is acrostic-based."""
    return "acrostic" in name.lower()


def is_whitespace_algorithm(name: str) -> bool:
    """Check if an algorithm is whitespace-based."""
    return name.startswith("whitespace")


__all__ = [
    "StegoAlgorithm",
    "AlgorithmRegistry",
    "WordAcrostic",
    "SentenceAcrostic",
    "SecondWordAcrostic",
    "SentenceBoundaryAcrostic",
    "FirstWordSentence",
    "WhitespaceBitsStego",
    "ZeroWidthStego",
    "get_algorithm",
    "list_algorithms",
    "is_acrostic_algorithm",
    "is_whitespace_algorithm",
]

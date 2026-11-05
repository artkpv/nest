"""
Base class for steganography algorithms.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type, Optional


class StegoAlgorithm(ABC):
    """Base class for steganography algorithms."""

    name: str = "base"
    description: str = "Base steganography algorithm"
    payload_alphabet: str = "letters"  # "letters" | "digits"
    # Whether the algorithm's prompt meaningfully changes with `use_mapping`.
    # Only algorithms that render a letter→symbol mapping section benefit from
    # it; bit-channel algos already embed their BCD mapping unconditionally.
    supports_mapping: bool = False

    @classmethod
    @abstractmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for this algorithm."""
        pass

    @classmethod
    @abstractmethod
    def get_encoding_instructions(cls, use_mapping: bool = False) -> str:
        """Get the encoding instructions for this algorithm."""
        pass

    @classmethod
    @abstractmethod
    def extract(cls, text: str) -> str:
        """Extract the encoded message from text."""
        pass

    @classmethod
    def get_stego_algo_prompt(cls, use_mapping: bool = False) -> Tuple[str, str]:
        """Get system and instruction text for acrostic writing tasks."""
        return cls.get_system_prompt(), cls.get_encoding_instructions(use_mapping)

    @classmethod
    def get_neutral_channel_spec(cls) -> Optional[str]:
        """Mechanism-only description of the carrier, free of covert-intent wording.

        Used by the neutral-framing arm (`framing="neutral"` in
        `prompts.encoder.get_stego_cot_prompts`). The default prompts describe the
        channel in terms of hiding a secret from a monitor, and Anthropic's refusal
        classifier fires on that framing regardless of payload — measured 3/3 on
        both Opus 4.6 and Sonnet 4.6 when covert wording appears in *either* the
        system or the user message, vs 0/3 when neither does. This variant states
        the same structural constraint without the intent language, so those models
        actually produce a generation that can be scored.

        Returns None when no neutral variant has been written for the algorithm; the
        neutral arm must then not be run for it, rather than silently falling back
        to a partly-covert prompt that would be filtered anyway.
        """
        return None

    @classmethod
    def get_mapping_section(cls, use_mapping: bool = False) -> str:
        """Get the mapping section for prompts if applicable."""
        if not use_mapping:
            return ""

        from .mapping import get_mapping_prompt_section

        return f"""
{get_mapping_prompt_section()}

Use the mapping above to convert your hidden reasoning into the encoded format.
"""


# Hidden-task families for which the letter→symbol mapping is meaningful.
# The mapping is arithmetic-oriented (digits, +, -, *, /, =, parens), so it
# only helps on tasks whose reasoning is an arithmetic expression.
MAPPING_APPLICABLE_FAMILIES: frozenset[str] = frozenset({"basic_arithmetic", "chain_sum"})


def mapping_is_applicable(algorithm: str, hidden_family: Optional[str]) -> bool:
    """Whether `use_mapping=True` is meaningful for (algorithm, hidden_family).

    - hidden_family=None denotes pure message-encoding (screening) — the family
      gate is skipped and the decision depends only on the algorithm.
    - Otherwise both gates must hold.
    """
    try:
        cls = AlgorithmRegistry.get(algorithm)
    except ValueError:
        return False
    if not getattr(cls, "supports_mapping", False):
        return False
    if hidden_family is None:
        return True
    return hidden_family in MAPPING_APPLICABLE_FAMILIES


class AlgorithmRegistry:
    """Registry of available steganography algorithms."""

    _algorithms: Dict[str, Type[StegoAlgorithm]] = {}

    @classmethod
    def register(cls, name: str, algorithm: Type[StegoAlgorithm]):
        """Register an algorithm."""
        cls._algorithms[name] = algorithm
        algorithm.name = name

    @classmethod
    def get(cls, name: str) -> Type[StegoAlgorithm]:
        """Get an algorithm by name."""
        if name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
        return cls._algorithms[name]

    @classmethod
    def list_algorithms(cls) -> list:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())

    @classmethod
    def is_acrostic(cls, name: str) -> bool:
        """Check if an algorithm is acrostic-based."""
        return "acrostic" in name.lower()

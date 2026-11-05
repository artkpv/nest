"""
Unified Inspect-based Steganography Evaluation Framework.

This package provides a single configurable task factory that generates
Inspect AI tasks for evaluating LLM steganography capabilities.
"""

from .tasks import stego_cot_eval

__all__ = ["stego_cot_eval"]

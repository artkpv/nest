"""Custom Inspect model providers for this project.

Importing this package registers the providers with Inspect's model registry,
so `stego_inspect/tasks.py` imports it unconditionally.
"""

from stego_inspect.providers.prefill_openrouter import PrefillOpenRouterAPI

__all__ = ["PrefillOpenRouterAPI"]

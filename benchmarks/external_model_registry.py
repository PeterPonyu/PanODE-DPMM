"""Backward-compatible shim for the canonical external baseline registry.

This module used to carry a second copy of the external benchmark registry.
The canonical implementation now lives in ``eval_lib.baselines.registry``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "benchmarks.external_model_registry is deprecated; use eval_lib.baselines.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

from eval_lib.baselines.registry import EXTERNAL_MODELS, MODEL_GROUPS, list_external_models

__all__ = ["EXTERNAL_MODELS", "MODEL_GROUPS", "list_external_models"]


if __name__ == "__main__":
    list_external_models()

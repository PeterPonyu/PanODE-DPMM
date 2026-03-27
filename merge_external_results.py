#!/usr/bin/env python3
"""Backward-compatible entry point for merging external benchmark groups.

Prefer `python -m experiments.merge_external_results` for new workflows.
This wrapper is kept so older local scripts still work.
"""

from __future__ import annotations

import warnings

from experiments.merge_external_results import main

warnings.warn(
    "merge_external_results.py is deprecated; use `python -m experiments.merge_external_results`.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    main()

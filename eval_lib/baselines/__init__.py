"""eval_lib.baselines — External baseline model wrappers and registry.

Submodules
----------
registry    EXTERNAL_MODELS dict mapping model names → factory + params
models/     BaseModel interface + 12 model wrapper implementations
"""

from .models import BaseModel
from .registry import (
    CLASSICAL_BASELINES,
    DEEP_GRAPH_BASELINES,
    EXTERNAL_MODELS,
    MODEL_TAXONOMY,
    list_external_models,
)

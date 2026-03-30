"""eval_lib.metrics — Latent-space and dimensionality-reduction metric evaluators.

Submodules
----------
dre     Dimensionality Reduction Evaluator (UMAP / t-SNE fidelity)
drex    Extended DR metrics (trustworthiness, continuity, …)
lse     Latent Space Evaluator (spectral decay, anisotropy, …)
lsex    Extended latent-space metrics (2-hop connectivity, curvature, …)
battery Unified metric battery (compute_metrics, METRIC_COLUMNS, diagnostics)
"""

from .battery import (
    METRIC_COLUMNS,
    METRIC_GROUPS,
    PUBLICATION_METRIC_GROUPS,
    DataSplitter,
    compute_latent_diagnostics,
    compute_metrics,
    convergence_diagnostics,
)
from .dre import evaluate_dimensionality_reduction
from .drex import evaluate_extended_dimensionality_reduction
from .lse import evaluate_single_cell_latent_space
from .lsex import evaluate_extended_latent_space

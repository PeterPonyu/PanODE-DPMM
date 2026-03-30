"""eval_lib.viz — Publication-quality figure generation from experiment results.

Submodules
----------
rea      RigorousExperimentalAnalyzer — statistical comparison figures
loss     Training / validation loss curve visualisation
layout   Centralised layout constants and adaptive sizing helpers
"""

from .layout import (  # noqa: F401
    FIG_WIDTH_PER_METRIC,
    MAX_HEIGHT_TO_WIDTH,
    MAX_METHODS_PER_FIGURE,
    MIN_HEIGHT_TO_WIDTH,
    MIN_XTICK_FONTSIZE,
    MIN_XTICK_FONTSIZE_PER_GROUP,
    assert_no_label_overlap,
    clamp_aspect_ratio,
    clamp_xtick_fontsize,
    needs_method_split,
)
from .loss import plot_aggregated_loss, plot_training_curves  # noqa: F401
from .rea import RigorousExperimentalAnalyzer, _apply_font, create_publication_figure  # noqa: F401

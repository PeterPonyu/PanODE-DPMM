"""Backward-compatible shim for legacy ``benchmarks.external_models`` imports.

The canonical external baseline implementations live in
``eval_lib.baselines.models``.  This package preserves the old import path
without maintaining a second full copy of the model library.
"""

from __future__ import annotations

from importlib import import_module
import sys

_CANONICAL_PACKAGE = "eval_lib.baselines.models"
_CANONICAL_MODULE = import_module(_CANONICAL_PACKAGE)

_COMPAT_SUBMODULES = [
    "base_model",
    "cellblast_model",
    "clear_model",
    "disentanglement_vae_model",
    "gmvae_model",
    "scalex_model",
    "scdac_model",
    "scdeepcluster_model",
    "scdhmap_model",
    "scdiffusion_model",
    "scgcc_model",
    "scgnn_model",
    "scsmd_model",
    "scvi_family_model",
    "sivae_model",
    "distributions",
    "distributions.utils",
    "distributions.EuclideanNormal",
    "distributions.EuclideanNormal.arguments",
    "distributions.EuclideanNormal.distribution",
    "distributions.EuclideanNormal.layers",
    "distributions.EuclideanNormal.prior",
    "distributions.HWNormal",
    "distributions.HWNormal.arguments",
    "distributions.HWNormal.distribution",
    "distributions.HWNormal.layers",
    "distributions.HWNormal.prior",
    "distributions.LearnablePGMNormal",
    "distributions.LearnablePGMNormal.arguments",
    "distributions.LearnablePGMNormal.distribution",
    "distributions.LearnablePGMNormal.layers",
    "distributions.LearnablePGMNormal.prior",
    "distributions.PGMNormal",
    "distributions.PGMNormal.arguments",
    "distributions.PGMNormal.distribution",
    "distributions.PGMNormal.layers",
    "distributions.PGMNormal.prior",
    "distributions.PoincareNormal",
    "distributions.PoincareNormal.arguments",
    "distributions.PoincareNormal.ars",
    "distributions.PoincareNormal.distribution",
    "distributions.PoincareNormal.hyperbolic_radius",
    "distributions.PoincareNormal.hyperbolic_uniform",
    "distributions.PoincareNormal.layers",
    "distributions.PoincareNormal.prior",
]


def _alias_submodule(relative_name: str) -> None:
    target = import_module(f"{_CANONICAL_PACKAGE}.{relative_name}")
    sys.modules[f"{__name__}.{relative_name}"] = target


for _submodule in _COMPAT_SUBMODULES:
    _alias_submodule(_submodule)

for _name in getattr(_CANONICAL_MODULE, "__all__", []):
    if hasattr(_CANONICAL_MODULE, _name):
        globals()[_name] = getattr(_CANONICAL_MODULE, _name)

__all__ = [
    _name for _name in getattr(_CANONICAL_MODULE, "__all__", [])
    if hasattr(_CANONICAL_MODULE, _name)
]

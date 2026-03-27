"""Compatibility tests for legacy external benchmark entry points."""

from __future__ import annotations

import importlib
import warnings

import pytest


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [
        ("benchmarks.external_models.base_model", "eval_lib.baselines.models.base_model"),
        ("benchmarks.external_models.scalex_model", "eval_lib.baselines.models.scalex_model"),
        (
            "benchmarks.external_models.distributions.PoincareNormal.layers",
            "eval_lib.baselines.models.distributions.PoincareNormal.layers",
        ),
    ],
)
def test_legacy_external_model_modules_alias_canonical_modules(
    legacy_name: str,
    canonical_name: str,
) -> None:
    legacy_module = importlib.import_module(legacy_name)
    canonical_module = importlib.import_module(canonical_name)
    assert legacy_module is canonical_module


def test_legacy_external_models_package_reexports_factories() -> None:
    legacy_pkg = importlib.import_module("benchmarks.external_models")
    canonical_pkg = importlib.import_module("eval_lib.baselines.models")
    assert legacy_pkg.create_scalex_model is canonical_pkg.create_scalex_model
    assert legacy_pkg.create_gmvae_model is canonical_pkg.create_gmvae_model


def test_legacy_external_registry_reexports_canonical_objects() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_registry = importlib.import_module("benchmarks.external_model_registry")

    canonical_registry = importlib.import_module("eval_lib.baselines.registry")
    assert legacy_registry.EXTERNAL_MODELS is canonical_registry.EXTERNAL_MODELS
    assert legacy_registry.MODEL_GROUPS is canonical_registry.MODEL_GROUPS
    assert legacy_registry.list_external_models is canonical_registry.list_external_models

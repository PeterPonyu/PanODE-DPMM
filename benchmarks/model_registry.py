"""Centralised model registry for all benchmark scripts.

Defines the canonical MODELS dict (7 variants across 2 architecture
families: DPMM, Pure-AE), series grouping constants,
ablation step definitions, and utility helpers.

Architecture Families:
- DPMM: AE + Dirichlet Process Mixture Model clustering
- Pure-AE: Deterministic autoencoder without any prior (independent)

Exports
-------
MODELS            : dict  — name → {class, params, series}
SERIES_GROUPS     : dict  — group tag → set of model names
ABLATION_STEPS    : dict  — series → [(model_name, step_label), …]
is_cuda_oom       : fn    — detect CUDA out-of-memory exceptions
"""

from models.dpmm_base import DPMMODEModel
from models.dpmm_flow_matching import DPMMFlowMatchingModel
from models.dpmm_contrastive import DPMMODEContrastiveModel
from models.dpmm_transformer import DPMMODETransformerModel

# Independent Pure-AE baselines (deterministic, no prior)
from models.pure_ae import PureAEModel, PureAETransformerModel, PureAEContrastiveModel


# ═══════════════════════════════════════════════════════════════════════════════
# 7 model configurations — tuned parameters from sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    # ── Pure-AE baselines (independent, no prior, no KL) ───────────────────
    'Pure-AE': {
        'class': PureAEModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dropout_rate': 0.2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'pure-ae',
    },
    'Pure-Transformer-AE': {
        'class': PureAETransformerModel,
        'params': {
            'latent_dim': 10,
            'd_model': 128,
            'decoder_dims': [128, 256],
            'dropout_rate': 0.2,
            'nhead': 4,
            'num_encoder_layers': 2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'pure-ae',
    },
    'Pure-Contrastive-AE': {
        'class': PureAEContrastiveModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dropout_rate': 0.2,
            'moco_weight': 1.0,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'pure-ae',
    },

    # ── DPMM series (AE backbone + DPMM clustering prior) ────────────────
    'DPMM-Base': {
        'class': DPMMODEModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.9,
            'dropout_rate': 0.2,               # from sensitivity: dropout=0.2 best
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,             # from sensitivity: wd=0 best for DPMM
            'fit_epochs': 1000,                # unified to 1000 epochs
        },
        'series': 'dpmm',
    },
    'DPMM-Transformer': {
        'class': DPMMODETransformerModel,
        'params': {
            'latent_dim': 10,
            'd_model': 128,
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.9,
            'dropout_rate': 0.2,
            'nhead': 4,
            'num_encoder_layers': 2,
            'dpmm_loss_weight': 0.1,
            'dpmm_anneal_epochs': 100,
            'var_reg_weight': 100.0,
            'var_reg_min': 0.1,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'dpmm',
    },
    'DPMM-Contrastive': {
        'class': DPMMODEContrastiveModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.9,
            'dropout_rate': 0.2,
            'moco_weight': 1.0,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'dpmm',
    },
    'DPMM-FM': {
        'class': DPMMFlowMatchingModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.8,
            'dropout_rate': 0.2,
            'flow_weight': 0.10,
            'flow_noise_scale': 0.5,
            'flow_after_dpmm': True,
            'flow_hidden_dims': [128, 128],
            'flow_t0': 0.8,
            'flow_smoothing': True,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'dpmm',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA OOM detection
# ═══════════════════════════════════════════════════════════════════════════════

def is_cuda_oom(exception) -> bool:
    """Return True if *exception* is a CUDA out-of-memory error."""
    msg = str(exception).lower()
    return "out of memory" in msg or "cuda" in msg and "alloc" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation step ordering (used by summary printers)
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_STEPS = {
    'dpmm': [
        ('DPMM-Base',           'Base AE+DPMM'),
        ('DPMM-Transformer',    '+Transformer'),
        ('DPMM-Contrastive',    '+Contrastive'),
        ('DPMM-FM',             '+Flow Matching'),
    ],
    'pure-ae': [
        ('Pure-AE',             'Base AE'),
        ('Pure-Transformer-AE', '+Transformer'),
        ('Pure-Contrastive-AE', '+Contrastive'),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Series grouping for --series CLI flag
# ═══════════════════════════════════════════════════════════════════════════════

SERIES_GROUPS = {
    'dpmm':     {k for k, v in MODELS.items() if v['series'] == 'dpmm'},
    'pure':     {k for k in MODELS if k.startswith('Pure-')},
    'pure-ae':  {k for k, v in MODELS.items() if v['series'] == 'pure-ae'},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Paper grouping — map each series to its output / comparison group
# Pure-AE is compared against DPMM
# ═══════════════════════════════════════════════════════════════════════════════

SERIES_TO_PAPER = {
    'dpmm':     'dpmm',
    'pure-ae':  'dpmm',
}


def paper_group(series_or_model: str) -> str:
    """Return the paper output group ('dpmm') for a series or model name."""
    if series_or_model in SERIES_TO_PAPER:
        return SERIES_TO_PAPER[series_or_model]
    # Lookup by model name
    if series_or_model in MODELS:
        return SERIES_TO_PAPER.get(MODELS[series_or_model]['series'], 'dpmm')
    return 'dpmm'

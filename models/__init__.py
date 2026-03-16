"""Model modules for DPMM and Pure-AE series.

Architecture Families:
- DPMM: AE + Dirichlet Process Mixture Model clustering
- Pure-AE: Deterministic autoencoder (no prior, no KL)

Variant Options:
- Base: MLP encoder
- Contrastive: MLP + MoCo contrastive learning
- Transformer: Multi-head projection encoder with O(batch_size) attention

Shared Modules:
- shared_modules.py: Common utilities (MLP, decoders)
- encoders.py: Unified encoder implementations (MLP, Transformer, Hybrid)
"""

# Shared modules
from .shared_modules import (
    weight_init,
    MLP,
    ResidualMLP,
    InformationBottleneck,
    MLPDecoder,
    TopicDecoder,
    SubgraphDataset,
    precompute_knn_graph,
    reparameterize,
    log_to_simplex)

# Encoder modules
from .encoders import (
    MultiHeadProjectionEncoder,
    MultiHeadProjectionLogisticNormalEncoder,
    HybridMLPAttentionEncoder,
    HybridMLPAttentionLogisticNormalEncoder,
    MLPEncoder,
    MLPLogisticNormalEncoder,
    create_encoder,
    create_topic_encoder)

# DPMM Models
from .dpmm_base import DPMMAutoEncoder
from .dpmm_flow_matching import DPMMFlowMatchingModel
from .dpmm_contrastive import DPMMContrastiveAutoEncoder
from .dpmm_transformer import DPMMTransformerAutoEncoder

# Pure-AE Models (no prior — DPMM ablation baseline)
from .pure_ae import PureAEModel, PureAETransformerModel, PureAEContrastiveModel

__all__ = [
    # Shared utilities
    'weight_init',
    'MLP',
    'ResidualMLP',
    'InformationBottleneck',
    'MLPDecoder',
    'TopicDecoder',
    'SubgraphDataset',
    'precompute_knn_graph',
    'reparameterize',
    'log_to_simplex',
    # Encoders
    'MultiHeadProjectionEncoder',
    'MultiHeadProjectionLogisticNormalEncoder',
    'HybridMLPAttentionEncoder',
    'HybridMLPAttentionLogisticNormalEncoder',
    'MLPEncoder',
    'MLPLogisticNormalEncoder',
    'create_encoder',
    'create_topic_encoder',
    # DPMM Models
    'DPMMAutoEncoder',
    'DPMMFlowMatchingModel',
    'DPMMContrastiveAutoEncoder',
    'DPMMTransformerAutoEncoder',
    # Pure-AE Models (no prior)
    'PureAEModel',
    'PureAETransformerModel',
    'PureAEContrastiveModel',
]

"""
Efficient Encoder Modules for Single-Cell Data.

This module provides encoder architectures for single-cell representation learning:

1. MLP ENCODER (Baseline):
   - Simple and fast
   - Good baseline performance
   - Optional residual connections

2. MULTI-HEAD PROJECTION TRANSFORMER (Recommended for attention):
   - Projects gene vector into multiple token embeddings
   - Self-attention learns relationships between projections
   - Prevents representation collapse through diverse projections
   - O(num_tokens²) complexity where num_tokens << n_genes

3. HYBRID MLP-ATTENTION (Balanced):
   - MLP backbone with multi-head attention refinement
   - Best of both worlds: MLP stability + attention expressiveness

4. GAT ENCODER (Graph Attention):
   - Graph Attention Network operating on a kNN cell graph
   - Leverages cell-cell neighbourhood structure via attention
   - Requires torch_geometric; gracefully unavailable otherwise
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal

from .shared_modules import weight_init, MLP, ResidualMLP

# Optional PyG dependency for GAT encoder
try:
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None


# =============================================================================
# MLP ENCODERS (Baseline)
# =============================================================================

class MLPEncoder(nn.Module):
    """Simple MLP encoder (fastest baseline)."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        use_residual: bool = False):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps

        if use_residual:
            self.encoder = ResidualMLP(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        else:
            layers = [input_dim] + [hidden_dim] * num_layers
            self.encoder = MLP(layers, dropout)

        if use_vae:
            self.mu_head = nn.Linear(hidden_dim, output_dim)
            self.var_head = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.apply(weight_init)
        # Re-apply after weight_init so the intended 0.5 is not overwritten
        if use_vae:
            nn.init.constant_(self.var_head.bias, 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h = self.encoder(x)

        if self.use_vae:
            mu = self.mu_head(h)
            var = F.softplus(self.var_head(h)) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_proj(h)
            return (z,)


# =============================================================================
# MULTI-HEAD PROJECTION TRANSFORMER
# =============================================================================

class MultiHeadProjectionEncoder(nn.Module):
    """
    Multi-Head Projection Transformer Encoder.

    Architecture:
    1. K parallel projections: gene vector -> K x d_model tokens
    2. Transformer self-attention across K tokens
    3. Learnable aggregation -> latent space

    This design prevents the single-token collapse problem.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 32,
        num_tokens: int = 8,
        use_vae: bool = False,
        var_eps: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.use_vae = use_vae
        self.var_eps = var_eps

        # Multiple projection heads
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout))
            for _ in range(num_tokens)
        ])

        # Token type embeddings
        self.token_embeddings = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Aggregation
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        if use_vae:
            self.mu_head = nn.Linear(d_model, output_dim)
            self.var_head = nn.Linear(d_model, output_dim)
        else:
            self.output_proj = nn.Linear(d_model, output_dim)

        self.apply(weight_init)
        # Re-apply after weight_init so the intended 0.5 is not overwritten
        if use_vae:
            nn.init.constant_(self.var_head.bias, 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size = x.size(0)

        # Create K tokens
        tokens = torch.stack([proj(x) for proj in self.projection_heads], dim=1)
        tokens = tokens + self.token_embeddings.expand(batch_size, -1, -1)

        # Transform
        tokens = self.transformer(tokens)

        # Aggregate
        query = self.aggregation_query.expand(batch_size, -1, -1)
        h, _ = self.cross_attention(query, tokens, tokens)
        h = h.squeeze(1)
        h = self.final_norm(h)

        if self.use_vae:
            mu = self.mu_head(h)
            var = F.softplus(self.var_head(h)) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_proj(h)
            return (z,)


# =============================================================================
# HYBRID MLP-ATTENTION ENCODER
# =============================================================================

class HybridMLPAttentionEncoder(nn.Module):
    """
    Hybrid MLP + Attention Encoder.

    Architecture:
    1. MLP backbone produces base embedding
    2. Multi-head attention refines across batch
    3. Residual connection preserves MLP stability
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        attention_weight: float = 0.3):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps
        self.attention_weight = attention_weight

        # MLP backbone
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))

        # Attention refinement
        self.attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output
        if use_vae:
            self.mu_head = nn.Linear(hidden_dim, output_dim)
            self.var_head = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.apply(weight_init)
        # Re-apply after weight_init so the intended 0.5 is not overwritten
        if use_vae:
            nn.init.constant_(self.var_head.bias, 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h_mlp = self.mlp_encoder(x)

        h_seq = h_mlp.unsqueeze(0)
        h_attn, _ = self.attention(h_seq, h_seq, h_seq)
        h_attn = h_attn.squeeze(0)

        h = h_mlp + self.attention_weight * (h_attn - h_mlp)
        h = self.attn_norm(h)

        if self.use_vae:
            mu = self.mu_head(h)
            var = F.softplus(self.var_head(h)) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_proj(h)
            return (z,)


# =============================================================================
# GAT (GRAPH ATTENTION) ENCODER
# =============================================================================

class GATConvEncoder(nn.Module):
    """Graph Attention Network encoder for cell-graph–aware representation learning.

    Architecture (following CCVGAE's polymorphic dispatch pattern):
      1. Stack of [GATConv → BatchNorm → ReLU → Dropout] layers
      2. Residual skip from first hidden layer to last
      3. Separate GATConv heads for mean/logvar (VAE) or a single linear output (AE)

    Requires ``torch_geometric``.  When the package is absent the
    ``create_encoder`` factory will raise ``ImportError`` rather than
    silently returning a broken object.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        use_residual: bool = True,
    ):
        super().__init__()
        if GATConv is None:
            raise ImportError(
                "GATConvEncoder requires torch_geometric. "
                "Install with: pip install torch-geometric"
            )
        self.use_vae = use_vae
        self.var_eps = var_eps
        self.use_residual = use_residual

        # Build GATConv stack
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer: input_dim → hidden_dim (multi-head, concat)
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
        self.dropouts.append(nn.Dropout(dropout))

        # Additional hidden layers: (hidden_dim * num_heads) → hidden_dim (concat)
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                        dropout=dropout, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
            self.dropouts.append(nn.Dropout(dropout))

        gat_out_dim = hidden_dim * num_heads

        # Output heads
        if use_vae:
            self.mu_head = GATConv(gat_out_dim, output_dim, heads=1,
                                   concat=False, dropout=dropout)
            self.var_head = GATConv(gat_out_dim, output_dim, heads=1,
                                    concat=False, dropout=dropout)
        else:
            self.output_conv = GATConv(gat_out_dim, output_dim, heads=1,
                                       concat=False, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor [N, input_dim]
            Node (cell) features.
        edge_index : LongTensor [2, E]
            COO-format edge indices of the cell graph.

        Returns
        -------
        (z,) or (z, mu, var)
        """
        residual = None
        h = x
        for i, (conv, bn, drop) in enumerate(
            zip(self.convs, self.bns, self.dropouts)
        ):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = drop(h)
            if self.use_residual and i == 0:
                residual = h

        if self.use_residual and residual is not None:
            h = h + residual

        if self.use_vae:
            mu = self.mu_head(h, edge_index)
            log_var = self.var_head(h, edge_index)
            var = F.softplus(log_var) + self.var_eps
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, var
        else:
            z = self.output_conv(h, edge_index)
            return (z,)


# =============================================================================
# ENCODER FACTORY
# =============================================================================

def create_encoder(
    encoder_type: Literal['mlp', 'transformer', 'hybrid', 'gat'],
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    use_vae: bool = False,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    var_eps: float = 1e-4,
    num_tokens: int = 8,
    attention_weight: float = 0.3,
    # GAT-specific
    gat_num_heads: int = 4,
    gat_use_residual: bool = True) -> nn.Module:
    """
    Factory function to create encoder.

    Args:
        encoder_type:
            - 'mlp': Simple MLP (fastest)
            - 'transformer': Multi-head projection transformer
            - 'hybrid': Hybrid MLP + attention
            - 'gat': Graph Attention Network (requires torch_geometric)
        input_dim: Number of input features (genes)
        output_dim: Latent dimension
        hidden_dim: Hidden dimension for MLP/hybrid
        use_vae: Whether to output (z, mu, var) or (z)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of layers
        dim_feedforward: Transformer feedforward dimension
        dropout: Dropout rate
        var_eps: Variance epsilon
        num_tokens: Number of projection heads (transformer)
        attention_weight: Attention blend ratio (hybrid)
        gat_num_heads: Number of GAT attention heads
        gat_use_residual: Whether to use residual skip in GAT

    Returns:
        Encoder module
    """
    if encoder_type == 'mlp':
        return MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_vae=use_vae,
            var_eps=var_eps)
    elif encoder_type == 'transformer':
        return MultiHeadProjectionEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim,
            num_tokens=num_tokens,
            use_vae=use_vae,
            var_eps=var_eps)
    elif encoder_type == 'hybrid':
        return HybridMLPAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            use_vae=use_vae,
            var_eps=var_eps,
            attention_weight=attention_weight)
    elif encoder_type == 'gat':
        if GATConv is None:
            raise ImportError(
                "encoder_type='gat' requires torch_geometric. "
                "Install with: pip install torch-geometric")
        return GATConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=gat_num_heads,
            dropout=dropout,
            use_vae=use_vae,
            var_eps=var_eps,
            use_residual=gat_use_residual)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'mlp', 'transformer', 'hybrid', or 'gat'.")

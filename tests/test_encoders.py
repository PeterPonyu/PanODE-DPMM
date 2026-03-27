"""Tests for encoder module — covers all encoder types including GATConvEncoder."""

import sys
import os
import pytest
import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.encoders import (
    MLPEncoder,
    MultiHeadProjectionEncoder,
    HybridMLPAttentionEncoder,
    create_encoder,
)

# GATConv is optional
try:
    from torch_geometric.nn import GATConv as _GATConv
    from models.encoders import GATConvEncoder
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ── Fixtures ──────────────────────────────────────────────────────────────────

INPUT_DIM = 200
HIDDEN_DIM = 64
OUTPUT_DIM = 16
BATCH_SIZE = 32
NUM_NODES = 64


def _random_edge_index(n_nodes: int, k: int = 5) -> torch.LongTensor:
    """Create a random kNN-like edge_index [2, E]."""
    rows, cols = [], []
    for i in range(n_nodes):
        neighbours = np.random.choice(n_nodes, size=k, replace=False)
        for j in neighbours:
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


# ── MLP Encoder ──────────────────────────────────────────────────────────────

class TestMLPEncoder:
    def test_ae_output_shape(self):
        enc = MLPEncoder(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, use_vae=False)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = enc(x)
        assert len(out) == 1
        assert out[0].shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_vae_output_shape(self):
        enc = MLPEncoder(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, use_vae=True)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        z, mu, var = enc(x)
        assert z.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert mu.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert var.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert (var > 0).all(), "Variance must be positive"

    def test_var_head_bias_init(self):
        enc = MLPEncoder(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, use_vae=True)
        assert torch.allclose(
            enc.var_head.bias,
            torch.full_like(enc.var_head.bias, 0.5),
        ), "var_head.bias should be 0.5 after init (not overwritten by weight_init)"


# ── Transformer Encoder ─────────────────────────────────────────────────────

class TestTransformerEncoder:
    def test_ae_output_shape(self):
        enc = MultiHeadProjectionEncoder(
            INPUT_DIM, d_model=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_tokens=4, use_vae=False)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = enc(x)
        assert len(out) == 1
        assert out[0].shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_vae_output_shape(self):
        enc = MultiHeadProjectionEncoder(
            INPUT_DIM, d_model=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_tokens=4, use_vae=True)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        z, mu, var = enc(x)
        assert z.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert (var > 0).all()

    def test_var_head_bias_init(self):
        enc = MultiHeadProjectionEncoder(
            INPUT_DIM, d_model=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_tokens=4, use_vae=True)
        assert torch.allclose(
            enc.var_head.bias,
            torch.full_like(enc.var_head.bias, 0.5),
        )


# ── Hybrid Encoder ───────────────────────────────────────────────────────────

class TestHybridEncoder:
    def test_ae_output_shape(self):
        enc = HybridMLPAttentionEncoder(
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, use_vae=False)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = enc(x)
        assert len(out) == 1
        assert out[0].shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_var_head_bias_init(self):
        enc = HybridMLPAttentionEncoder(
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, use_vae=True)
        assert torch.allclose(
            enc.var_head.bias,
            torch.full_like(enc.var_head.bias, 0.5),
        )


# ── GAT Encoder ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestGATConvEncoder:
    def test_ae_output_shape(self):
        enc = GATConvEncoder(
            INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_layers=2, num_heads=2, use_vae=False)
        x = torch.randn(NUM_NODES, INPUT_DIM)
        edge_index = _random_edge_index(NUM_NODES)
        out = enc(x, edge_index)
        assert len(out) == 1
        assert out[0].shape == (NUM_NODES, OUTPUT_DIM)

    def test_vae_output_shape(self):
        enc = GATConvEncoder(
            INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_layers=2, num_heads=2, use_vae=True)
        x = torch.randn(NUM_NODES, INPUT_DIM)
        edge_index = _random_edge_index(NUM_NODES)
        z, mu, var = enc(x, edge_index)
        assert z.shape == (NUM_NODES, OUTPUT_DIM)
        assert mu.shape == (NUM_NODES, OUTPUT_DIM)
        assert var.shape == (NUM_NODES, OUTPUT_DIM)
        assert (var > 0).all(), "Variance must be positive"

    def test_residual_changes_output(self):
        """Residual skip should make a difference vs no residual."""
        enc_res = GATConvEncoder(
            INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_layers=2, num_heads=2, use_vae=False, use_residual=True)
        enc_nores = GATConvEncoder(
            INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_layers=2, num_heads=2, use_vae=False, use_residual=False)
        # Copy weights to make them identical
        enc_nores.load_state_dict(enc_res.state_dict(), strict=False)

        x = torch.randn(NUM_NODES, INPUT_DIM)
        edge_index = _random_edge_index(NUM_NODES)
        z_res = enc_res(x, edge_index)[0]
        z_nores = enc_nores(x, edge_index)[0]
        # They should differ (residual adds skip connection)
        assert not torch.allclose(z_res, z_nores, atol=1e-5)

    def test_single_layer(self):
        enc = GATConvEncoder(
            INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_layers=1, num_heads=2, use_vae=False)
        x = torch.randn(NUM_NODES, INPUT_DIM)
        edge_index = _random_edge_index(NUM_NODES)
        out = enc(x, edge_index)
        assert out[0].shape == (NUM_NODES, OUTPUT_DIM)

    def test_gradient_flow(self):
        enc = GATConvEncoder(
            INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_layers=2, num_heads=2, use_vae=False)
        x = torch.randn(NUM_NODES, INPUT_DIM, requires_grad=True)
        edge_index = _random_edge_index(NUM_NODES)
        z = enc(x, edge_index)[0]
        loss = z.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_head_counts(self):
        for heads in [1, 2, 4, 8]:
            enc = GATConvEncoder(
                INPUT_DIM, hidden_dim=32, output_dim=OUTPUT_DIM,
                num_layers=2, num_heads=heads, use_vae=False)
            x = torch.randn(NUM_NODES, INPUT_DIM)
            edge_index = _random_edge_index(NUM_NODES)
            out = enc(x, edge_index)
            assert out[0].shape == (NUM_NODES, OUTPUT_DIM)


# ── Factory ──────────────────────────────────────────────────────────────────

class TestEncoderFactory:
    def test_mlp(self):
        enc = create_encoder('mlp', INPUT_DIM, OUTPUT_DIM, hidden_dim=HIDDEN_DIM)
        assert isinstance(enc, MLPEncoder)

    def test_transformer(self):
        enc = create_encoder('transformer', INPUT_DIM, OUTPUT_DIM, d_model=HIDDEN_DIM)
        assert isinstance(enc, MultiHeadProjectionEncoder)

    def test_hybrid(self):
        enc = create_encoder('hybrid', INPUT_DIM, OUTPUT_DIM, hidden_dim=HIDDEN_DIM)
        assert isinstance(enc, HybridMLPAttentionEncoder)

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_gat(self):
        enc = create_encoder('gat', INPUT_DIM, OUTPUT_DIM, hidden_dim=HIDDEN_DIM)
        assert isinstance(enc, GATConvEncoder)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_encoder('nonexistent', INPUT_DIM, OUTPUT_DIM)

    @pytest.mark.skipif(HAS_PYG, reason="Only test when torch_geometric is absent")
    def test_gat_import_error_without_pyg(self):
        with pytest.raises(ImportError, match="torch_geometric"):
            create_encoder('gat', INPUT_DIM, OUTPUT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

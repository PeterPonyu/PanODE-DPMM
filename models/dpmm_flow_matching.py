"""DPMM-FM: DPMM autoencoder with latent flow matching.

This variant keeps the strong DPMM-AE backbone and adds a latent-space
flow-matching head that learns a smooth transport from Gaussian noise to the
learned DPMM latent distribution.

Design
------
- backbone: same encoder/decoder + DPMM prior as ``DPMMODEModel``
- extra head: time-conditioned latent velocity field
- training: reconstruction + DPMM loss + flow-matching loss
- activation: flow loss can be delayed until DPMM fitting has started so the
  vector field regularizes the *clustered* latent space rather than the raw AE
  warmup manifold
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpmm_base import DPMMODEModel, MLP


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for scalar time inputs."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = int(embed_dim)
        if self.embed_dim < 2:
            raise ValueError("embed_dim must be >= 2")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.float()
        half = self.embed_dim // 2
        if half == 0:
            return t[:, None]
        freq = torch.exp(
            torch.linspace(0.0, -math.log(10000.0), half, device=t.device, dtype=t.dtype)
        )
        phase = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        if emb.shape[1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[1]))
        return emb


class LatentFlowField(nn.Module):
    """Time-conditioned latent velocity network."""

    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int = 32,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        features = [latent_dim + time_embed_dim] + list(hidden_dims or [128, 128]) + [latent_dim]
        self.net = MLP(
            features,
            hid_act="mish",
            out_act=None,
            norm=None,
            hid_norm="ln",
            drop=0.0,
            hid_drop=dropout,
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([z_t, t_emb], dim=1))


class DPMMFlowMatchingModel(DPMMODEModel):
    """DPMM-AE with a latent-space flow-matching regularizer."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_dims: Optional[list] = None,
        decoder_dims: Optional[list] = None,
        norm_type: str = "bn",
        dropout_rate: float = 0.1,
        dpmm_warmup_ratio: float = 0.8,
        dpmm_loss_weight: float = 1.0,
        dpmm_refit_interval: int = 10,
        n_components: int = 50,
        dpmm_loss_type: str = "kl",
        student_t_df: float = 3.0,
        mmd_bandwidth: float = 1.0,
        model_name: str = "DPMMFM",
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        use_vae: bool = False,
        kl_weight: float = 0.1,
        flow_weight: float = 0.10,
        flow_hidden_dims: Optional[list[int]] = None,
        flow_time_dim: int = 32,
        flow_noise_scale: float = 0.5,
        flow_after_dpmm: bool = True,
        flow_detach_target: bool = False,
        flow_dropout: float = 0.05,
        flow_integration_steps: int = 16,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            dpmm_warmup_ratio=dpmm_warmup_ratio,
            dpmm_loss_weight=dpmm_loss_weight,
            dpmm_refit_interval=dpmm_refit_interval,
            n_components=n_components,
            dpmm_loss_type=dpmm_loss_type,
            student_t_df=student_t_df,
            mmd_bandwidth=mmd_bandwidth,
            model_name=model_name,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            use_vae=use_vae,
            kl_weight=kl_weight,
            **kwargs,
        )
        self.flow_weight = float(flow_weight)
        self.flow_noise_scale = float(flow_noise_scale)
        self.flow_after_dpmm = bool(flow_after_dpmm)
        self.flow_detach_target = bool(flow_detach_target)
        self.flow_integration_steps = int(flow_integration_steps)
        self.flow_field = LatentFlowField(
            latent_dim=latent_dim,
            time_embed_dim=flow_time_dim,
            hidden_dims=flow_hidden_dims or [128, 128],
            dropout=flow_dropout,
        )

    def _flow_active(self) -> bool:
        if self.flow_weight <= 0:
            return False
        if self.flow_after_dpmm:
            return bool(self.dpmm_fitted)
        return True

    def _sample_flow_targets(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_target = latent.detach() if self.flow_detach_target else latent
        z_source = torch.randn_like(z_target) * self.flow_noise_scale
        t = torch.rand(z_target.size(0), device=z_target.device)
        z_t = (1.0 - t[:, None]) * z_source + t[:, None] * z_target
        velocity_target = z_target - z_source
        return z_t, t, velocity_target

    def compute_flow_loss(self, latent: torch.Tensor) -> torch.Tensor:
        z_t, t, velocity_target = self._sample_flow_targets(latent)
        velocity_pred = self.flow_field(z_t, t)
        return F.mse_loss(velocity_pred, velocity_target)

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_loss(x, outputs, **kwargs)
        flow_loss = torch.tensor(0.0, device=x.device)
        if self._flow_active():
            flow_loss = self.compute_flow_loss(outputs["latent"])
        loss_dict["flow_loss"] = flow_loss
        loss_dict["total_loss"] = loss_dict["total_loss"] + self.flow_weight * flow_loss
        return loss_dict

    @torch.no_grad()
    def sample_latent_prior(self, n_samples: int, device: str | torch.device = "cpu", steps: Optional[int] = None) -> torch.Tensor:
        """Sample from the learned latent flow by Euler integration from Gaussian noise."""
        self.eval()
        device = torch.device(device)
        self.to(device)
        n_steps = int(steps or self.flow_integration_steps)
        z = torch.randn(int(n_samples), self.latent_dim, device=device) * self.flow_noise_scale
        times = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        for idx in range(n_steps):
            t = torch.full((n_samples,), float(times[idx]), device=device)
            dt = float(times[idx + 1] - times[idx])
            z = z + dt * self.flow_field(z, t)
        return z


def create_dpmm_fm_model(input_dim: int, latent_dim: int = 32, **kwargs) -> DPMMFlowMatchingModel:
    """Factory for the DPMM-FM model."""
    return DPMMFlowMatchingModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)

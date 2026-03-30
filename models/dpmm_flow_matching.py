"""DPMM-FM: DPMM autoencoder with latent flow matching.

This variant keeps the strong DPMM-AE backbone and adds a latent-space
flow-matching head that learns a smooth transport from Gaussian noise to the
learned DPMM latent distribution.

Design
------
- backbone: same encoder/decoder + DPMM prior as ``DPMMODEModel``
- extra head: time-conditioned latent velocity field (conditional OT FM)
- training: reconstruction + DPMM loss + flow-matching loss
- activation: flow loss can be delayed until DPMM fitting has started so the
  vector field regularises the *clustered* latent space rather than the raw AE
  warmup manifold

Two-stage motivation (brainstorm)
---------------------------------
The FM learns the complete noise-to-data transport path t ∈ [0, 1]:

    z_t = (1-t)·ε + t·z_real,   v*(z_t, t) = z_real - ε

At *inference* the FM is used as a **latent smoother** rather than a pure
generative prior:

1. Encode a real cell:  z_real = encoder(x)
2. Perturb at a high time-step t₀ ∈ [0.7, 0.9]:
       z_{t₀} = (1 - t₀)·ε + t₀·z_real
3. Euler-integrate the learned velocity field from t₀ → 1.0
4. The result z_smooth is projected back onto the data manifold

``t₀`` controls the smoothing intensity:
  - t₀ → 1.0: near-identity (only posterior noise removed)
  - t₀ → 0.0: full generation from noise (cell identity lost)

When ``flow_detach_target=True`` the encoder is effectively frozen during
FM training (two-stage / decoupled mode).  When ``False`` (default) encoder
and FM co-adapt (joint mode).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpmm_base import MLP, DPMMODEModel


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
        hidden_dims: list[int] | None = None,
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
        encoder_dims: list | None = None,
        decoder_dims: list | None = None,
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
        bottleneck_dim: int | None = None,
        use_vae: bool = False,
        kl_weight: float = 0.1,
        flow_weight: float = 0.10,
        flow_hidden_dims: list[int] | None = None,
        flow_time_dim: int = 32,
        flow_noise_scale: float = 0.5,
        flow_after_dpmm: bool = True,
        flow_detach_target: bool = False,
        flow_dropout: float = 0.05,
        flow_integration_steps: int = 16,
        flow_t0: float = 0.8,
        flow_smoothing: bool = True,
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
        self.flow_t0 = float(flow_t0)
        self.flow_smoothing = bool(flow_smoothing)
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

    def compute_loss(self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        loss_dict = super().compute_loss(x, outputs, **kwargs)
        flow_loss = torch.tensor(0.0, device=x.device)
        if self._flow_active():
            flow_loss = self.compute_flow_loss(outputs["latent"])
        loss_dict["flow_loss"] = flow_loss
        loss_dict["total_loss"] = loss_dict["total_loss"] + self.flow_weight * flow_loss
        return loss_dict

    @torch.no_grad()
    def sample_latent_prior(self, n_samples: int, device: str | torch.device = "cpu", steps: int | None = None) -> torch.Tensor:
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

    # ------------------------------------------------------------------
    # Latent smoothing (brainstorm §B: partial-path FM inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def smooth_latent(
        self,
        z_real: torch.Tensor,
        t0: float | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Smooth encoder latents by partial FM integration from *t0* → 1.0.

        Instead of generating new samples from noise (``sample_latent_prior``),
        this method starts from an existing encoder output perturbed to
        time-step *t0* on the OT interpolation path and integrates only the
        remaining segment of the velocity field.

        Parameters
        ----------
        z_real : Tensor [N, latent_dim]
            Raw encoder output (already on the correct device).
        t0 : float, optional
            Starting time-step.  Higher → less smoothing.
            Default: ``self.flow_t0`` (set at init, typically 0.8).
        steps : int, optional
            Number of Euler steps for the *t0 → 1* segment.
            Default: ``self.flow_integration_steps``.

        Returns
        -------
        z_smooth : Tensor [N, latent_dim]
        """
        t0 = float(t0 if t0 is not None else self.flow_t0)
        n_steps = int(steps or self.flow_integration_steps)
        n = z_real.size(0)
        device = z_real.device

        # Perturb to the OT midpoint at t0:
        #   z_{t0} = (1 - t0) * ε + t0 * z_real
        eps = torch.randn_like(z_real) * self.flow_noise_scale
        z = (1.0 - t0) * eps + t0 * z_real

        # Euler integration from t0 → 1.0
        times = torch.linspace(t0, 1.0, n_steps + 1, device=device)
        for idx in range(n_steps):
            t_vec = torch.full((n,), float(times[idx]), device=device)
            dt = float(times[idx + 1] - times[idx])
            z = z + dt * self.flow_field(z, t_vec)
        return z

    # ------------------------------------------------------------------
    # Override extract_latent to optionally apply FM smoothing
    # ------------------------------------------------------------------

    def extract_latent(
        self,
        data_loader,
        device: str = "cuda",
        return_reconstructions: bool = False,
        smooth: bool | None = None,
        t0: float | None = None,
        **kwargs,
    ):
        """Extract latent representations, optionally FM-smoothed.

        Parameters
        ----------
        smooth : bool or None
            If ``True``, apply ``smooth_latent`` to every batch.
            If ``None`` (default), uses ``self.flow_smoothing``.
        t0 : float or None
            Override for smoothing start time.
        """
        use_smooth = smooth if smooth is not None else self.flow_smoothing
        # If smoothing is off or FM has not been trained, fall back to base
        if not use_smooth or not self._flow_active():
            return super().extract_latent(
                data_loader, device=device,
                return_reconstructions=return_reconstructions, **kwargs)

        self.eval()
        self.to(device)
        latents, recons = [], []
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                z_raw = self.encode(x, **batch_kwargs)
                z_smooth = self.smooth_latent(z_raw, t0=t0)
                latents.append(z_smooth.cpu().numpy())
                if return_reconstructions:
                    recons.append(self.decode(z_smooth).cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result


def create_dpmm_fm_model(input_dim: int, latent_dim: int = 32, **kwargs) -> DPMMFlowMatchingModel:
    """Factory for the DPMM-FM model."""
    return DPMMFlowMatchingModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)

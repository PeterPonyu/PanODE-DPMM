"""
Pure AE: Standard Autoencoder with no prior regularization.

Fully independent from Topic and DPMM architectures.
No KL divergence, no Dirichlet prior, no DPMM clustering, no simplex.
Pure reconstruction autoencoder with MSE loss.

Three variants:
- PureAEModel: MLP encoder/decoder
- PureAETransformerModel: Multi-head projection transformer encoder
- PureAEContrastiveModel: MLP encoder + MoCo contrastive learning
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, Literal

try:
    from .base_model import BaseModel
except ImportError:
    from utils.base_model import BaseModel


# =============================================================================
# Weight Initialization
# =============================================================================

def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


# =============================================================================
# Building Blocks
# =============================================================================

def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class _Layer1D(nn.Module):
    """Normalization + Activation + Dropout layer"""
    def __init__(self, dim: int, norm: Optional[str] = None,
                 act: Optional[str] = None, drop: float = 0.0):
        super().__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if act:
            layers.append(_act(act))
        if drop and drop > 0:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """Configurable MLP with flexible normalization, activation, and dropout."""
    def __init__(
        self,
        features: list,
        hid_act: str = "mish",
        out_act: Optional[str] = None,
        norm: Optional[str] = None,
        hid_norm: Optional[str] = None,
        drop: float = 0.0,
        hid_drop: float = 0.0):
        super().__init__()
        layers = []
        for i in range(1, len(features)):
            is_last = i == len(features) - 1
            cur_norm = norm if is_last else hid_norm
            cur_act = out_act if is_last else hid_act
            cur_drop = drop if is_last else hid_drop
            layers.append(nn.Linear(features[i - 1], features[i]))
            if cur_norm or cur_act or cur_drop:
                layers.append(_Layer1D(features[i], norm=cur_norm, act=cur_act, drop=cur_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Deterministic Encoder (no mu/var split)
# =============================================================================

class DeterministicEncoder(nn.Module):
    """MLP encoder that directly outputs z (no stochastic sampling)."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        encoder_dims: Optional[list] = None,
        norm: str = "bn",
        drop: float = 0.2):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [256, 128]
        self.encoder = MLP(
            [input_dim] + encoder_dims + [latent_dim],
            hid_act="mish", norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# =============================================================================
# Transformer Deterministic Encoder
# =============================================================================

class TransformerDeterministicEncoder(nn.Module):
    """Multi-head projection transformer encoder, outputs z directly."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens

        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout))
            for _ in range(num_tokens)
        ])
        self.token_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.aggregation_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        tokens = torch.stack([proj(x) for proj in self.projection_heads], dim=1)
        tokens = tokens + self.token_embeddings.expand(batch_size, -1, -1)
        tokens = self.transformer(tokens)
        query = self.aggregation_query.expand(batch_size, -1, -1)
        h, _ = self.cross_attention(query, tokens, tokens)
        h = h.squeeze(1)
        h = self.final_norm(h)
        return self.output_proj(h)


# =============================================================================
# Data Augmentation (for contrastive variant)
# =============================================================================

class DataAugmentation(nn.Module):
    """Augmentation strategies for single-cell data (feature dropout + noise)."""
    def __init__(
        self,
        noise_prob: float = 0.2,
        noise_std: float = 0.1,
        mask_prob: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        feature_dropout: float = 0.2):
        super().__init__()
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.scale_range = scale_range
        self.feature_dropout = feature_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        device = x.device
        x_aug = x.clone()
        if self.feature_dropout > 0:
            mask = torch.bernoulli(
                torch.ones(batch_size, n_features, device=device) * (1 - self.feature_dropout)
            )
            x_aug = x_aug * mask
        if self.noise_prob > 0 and torch.rand(1).item() < self.noise_prob:
            x_aug = x_aug + torch.randn_like(x_aug) * self.noise_std
        if self.mask_prob > 0:
            mask = torch.rand(batch_size, n_features, device=device) > self.mask_prob
            x_aug = x_aug * mask.float()
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.empty(batch_size, 1, device=device).uniform_(*self.scale_range)
            x_aug = x_aug * scale
        return x_aug


# =============================================================================
# Momentum Contrast (MoCo)
# =============================================================================

class MomentumContrast(nn.Module):
    """MoCo module: query/key projectors + memory queue + InfoNCE."""
    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.2):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.query_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embedding_dim))
        self.key_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embedding_dim))
        for p_q, p_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(embedding_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.apply(weight_init)

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            part1 = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:part1].T
            self.queue[:, :batch_size - part1] = keys[part1:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, z_query: torch.Tensor, z_key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.query_projector(z_query), dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.key_projector(z_key), dim=1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels


# =============================================================================
# PureAEModel — Base variant
# =============================================================================

class PureAEModel(BaseModel):
    """Pure Autoencoder: no KL, no clustering prior.

    Architecture:
    - Encoder: MLP → z (deterministic)
    - Decoder: MLP → reconstruction
    - Loss: MSE reconstruction only

    No VAE component, no DPMM, no Topic prior.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        encoder_dims: Optional[list] = None,
        decoder_dims: Optional[list] = None,
        dropout_rate: float = 0.2,
        norm_type: str = "bn",
        model_name: str = "PureAE",
        # Fit-specific (popped by train_and_evaluate)
        fit_lr: float = 1e-3,
        fit_weight_decay: float = 0,
        fit_epochs: int = 1000):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_dims or [256, 128],
            model_name=model_name)
        if encoder_dims is None:
            encoder_dims = [256, 128]
        if decoder_dims is None:
            decoder_dims = [128, 256]

        self.recon_loss_fn = nn.MSELoss()

        # Encoder: deterministic
        self.encoder_net = DeterministicEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            norm=norm_type,
            drop=dropout_rate)

        # Decoder
        self.decoder_net = MLP(
            [latent_dim] + decoder_dims + [input_dim],
            hid_act="mish", norm=norm_type, hid_norm=norm_type, hid_drop=dropout_rate)
        self.decoder_net.apply(weight_init)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        z = self.encoder_net(x)
        x_hat = self.decoder_net(z)
        return {
            "reconstruction": x_hat,
            "latent": z,
        }

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        recon = self.recon_loss_fn(outputs["reconstruction"], x)
        return {
            "total_loss": recon,
            "recon_loss": recon,
        }

    def extract_latent(self, data_loader, device='cuda',
                       return_reconstructions: bool = False, **kwargs):
        self.eval()
        self.to(device)
        latents, recons = [], []
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                z = self.encode(x)
                latents.append(z.cpu().numpy())
                if return_reconstructions:
                    recons.append(self.decode(z).cpu().numpy())
        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1000,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 50,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 0,
        **kwargs):
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        patience_counter = 0
        train_losses, recon_losses, contrastive_losses = [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs)
                loss = loss_dict["total_loss"]
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                continue
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            recon_losses.append(avg_loss)

            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )
            if do_print:
                print(f"Epoch {epoch+1:3d}/{epochs} [PureAE] | "
                      f"Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
        }


# =============================================================================
# PureAETransformerModel — Transformer variant
# =============================================================================

class PureAETransformerModel(BaseModel):
    """Pure AE with multi-head projection transformer encoder.

    Deterministic transformer encoder + MLP decoder.
    No KL, no prior, no clustering.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        decoder_dims: Optional[list] = None,
        dropout_rate: float = 0.2,
        num_tokens: int = 8,
        model_name: str = "PureTransformerAE",
        fit_lr: float = 1e-3,
        fit_weight_decay: float = 0,
        fit_epochs: int = 1000):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[d_model],
            model_name=model_name)
        self.recon_loss_fn = nn.MSELoss()

        # Transformer encoder (deterministic)
        self.encoder_net = TransformerDeterministicEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            num_tokens=num_tokens)

        # MLP decoder
        if decoder_dims is None:
            decoder_dims = [128, 256]
        self.decoder_net = MLP(
            [latent_dim] + decoder_dims + [input_dim],
            hid_act="mish", norm="bn", hid_norm="bn", hid_drop=dropout_rate)
        self.decoder_net.apply(weight_init)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        z = self.encoder_net(x)
        x_hat = self.decoder_net(z)
        return {
            "reconstruction": x_hat,
            "latent": z,
        }

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        recon = self.recon_loss_fn(outputs["reconstruction"], x)
        return {
            "total_loss": recon,
            "recon_loss": recon,
        }

    def extract_latent(self, data_loader, device='cuda',
                       return_reconstructions: bool = False, **kwargs):
        self.eval()
        self.to(device)
        latents, recons = [], []
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                z = self.encode(x)
                latents.append(z.cpu().numpy())
                if return_reconstructions:
                    recons.append(self.decode(z).cpu().numpy())
        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1000,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 50,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 0,
        **kwargs):
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        patience_counter = 0
        train_losses, recon_losses, contrastive_losses = [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs)
                loss = loss_dict["total_loss"]
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                continue
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            recon_losses.append(avg_loss)

            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )
            if do_print:
                print(f"Epoch {epoch+1:3d}/{epochs} [PureTransAE] | "
                      f"Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
        }


# =============================================================================
# PureAEContrastiveModel — Contrastive variant
# =============================================================================

class PureAEContrastiveModel(BaseModel):
    """Pure AE + MoCo contrastive learning.

    Deterministic encoder with additional InfoNCE contrastive loss.
    No KL, no prior, no clustering.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        encoder_dims: Optional[list] = None,
        decoder_dims: Optional[list] = None,
        dropout_rate: float = 0.2,
        norm_type: str = "bn",
        moco_weight: float = 1.0,
        moco_queue_size: int = 4096,
        moco_momentum: float = 0.999,
        moco_temperature: float = 0.2,
        moco_embedding_dim: int = 128,
        model_name: str = "PureContrastiveAE",
        fit_lr: float = 1e-3,
        fit_weight_decay: float = 0,
        fit_epochs: int = 1000):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_dims or [256, 128],
            model_name=model_name)
        if encoder_dims is None:
            encoder_dims = [256, 128]
        if decoder_dims is None:
            decoder_dims = [128, 256]

        self.moco_weight = moco_weight
        self.recon_loss_fn = nn.MSELoss()

        # Encoder (query)
        self.encoder_net = DeterministicEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            norm=norm_type,
            drop=dropout_rate)

        # Momentum encoder (key)
        self.momentum_encoder = DeterministicEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            norm=norm_type,
            drop=dropout_rate)
        for p_q, p_k in zip(self.encoder_net.parameters(), self.momentum_encoder.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # Decoder
        self.decoder_net = MLP(
            [latent_dim] + decoder_dims + [input_dim],
            hid_act="mish", norm=norm_type, hid_norm=norm_type, hid_drop=dropout_rate)
        self.decoder_net.apply(weight_init)

        # MoCo
        self.moco = MomentumContrast(
            latent_dim=latent_dim,
            embedding_dim=moco_embedding_dim,
            queue_size=moco_queue_size,
            momentum=moco_momentum,
            temperature=moco_temperature)

        # Augmentation
        self.augmentation = DataAugmentation(
            noise_prob=0.2, noise_std=0.1, mask_prob=0.1,
            feature_dropout=0.2)

        self.contrastive_loss_fn = nn.CrossEntropyLoss()

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decoder_net(z)

    @torch.no_grad()
    def _momentum_update_encoder(self):
        m = self.moco.momentum
        for p_q, p_k in zip(self.encoder_net.parameters(), self.momentum_encoder.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        z = self.encoder_net(x)
        x_hat = self.decoder_net(z)
        result = {
            "reconstruction": x_hat,
            "latent": z,
        }

        if self.training:
            x_aug = self.augmentation(x)
            with torch.no_grad():
                self._momentum_update_encoder()
                z_key = self.momentum_encoder(x_aug)
            logits, labels = self.moco(z, z_key)
            result["contrastive_logits"] = logits
            result["contrastive_labels"] = labels

        return result

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        recon = self.recon_loss_fn(outputs["reconstruction"], x)
        loss_dict = {"recon_loss": recon}
        total = recon

        if "contrastive_logits" in outputs:
            contra = self.contrastive_loss_fn(
                outputs["contrastive_logits"], outputs["contrastive_labels"]
            )
            loss_dict["contrastive_loss"] = contra
            total = total + self.moco_weight * contra

        loss_dict["total_loss"] = total
        return loss_dict

    def extract_latent(self, data_loader, device='cuda',
                       return_reconstructions: bool = False, **kwargs):
        self.eval()
        self.to(device)
        latents, recons = [], []
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                z = self.encode(x)
                latents.append(z.cpu().numpy())
                if return_reconstructions:
                    recons.append(self.decode(z).cpu().numpy())
        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1000,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 50,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 0,
        **kwargs):
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        patience_counter = 0
        train_losses, recon_losses, contrastive_losses = [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.train()
            epoch_loss = epoch_recon = epoch_contra = 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs)
                loss = loss_dict["total_loss"]
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                if "contrastive_loss" in loss_dict:
                    epoch_contra += loss_dict["contrastive_loss"].item()
                n_batches += 1

            if n_batches == 0:
                continue
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_contra = epoch_contra / n_batches
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            contrastive_losses.append(avg_contra)

            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )
            if do_print:
                print(f"Epoch {epoch+1:3d}/{epochs} [PureContrAE] | "
                      f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Contra: {avg_contra:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
            "contrastive_loss": contrastive_losses,
        }

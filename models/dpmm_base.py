"""
DPMMODE: Autoencoder with DPMM Clustering

Single-phase training:
- Phase 1: Train AE/VAE with DPMM clustering (fit method)
"""
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Literal
from sklearn.mixture import BayesianGaussianMixture
from scipy.sparse import issparse
try:
    from .base_model import BaseModel
except ImportError:
    from utils.base_model import BaseModel


def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)

def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "mish":
        return nn.Mish()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")

class _Layer1D(nn.Module):
    """Normalization + Activation + Dropout layer"""
    def __init__(self, dim: int, norm: Optional[str] = None, act: Optional[str] = None, drop: float = 0.0):
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
    """Configurable MLP with flexible normalization, activation, and dropout"""
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

class InformationBottleneck(nn.Module):
    """Compress and reconstruct latent space"""
    def __init__(self, latent_dim: int, bottleneck_dim: int, norm: str = "bn", drop: float = 0.1):
        super().__init__()
        self.compress = MLP([latent_dim, latent_dim//2, bottleneck_dim], 
                           norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)
        self.expand = MLP([bottleneck_dim, latent_dim//2, latent_dim], 
                         norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)
    
    def forward(self, z):
        z_bottleneck = self.compress(z)
        z_reconstructed = self.expand(z_bottleneck)
        return z_bottleneck, z_reconstructed

class DPMMAutoEncoder(nn.Module):
    """Autoencoder with optional VAE and bottleneck components."""
    def __init__(
        self, 
        input_dim: int, 
        encoder_dims: list, 
        latent_dim: int, 
        decoder_dims: list, 
        norm: str, 
        drop: float, 
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        **kwargs):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps
        self.latent_dim = latent_dim
        
        # Encoder: shared backbone + VAE heads (mu, var) or direct embedding
        if use_vae:
            # VAE mode: encoder backbone + mu/var heads
            self.encoder_backbone = MLP([input_dim] + encoder_dims, 
                                        norm=norm, hid_norm=norm, hid_drop=drop, out_act="mish")
            self.mu_head = nn.Linear(encoder_dims[-1], latent_dim)
            self.var_head = nn.Linear(encoder_dims[-1], latent_dim)
        else:
            # AE mode: direct encoder
            self.encoder = MLP([input_dim] + encoder_dims + [latent_dim], 
                              norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)
        
        self.decoder = MLP([latent_dim] + decoder_dims + [input_dim], 
                          norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)
        
        self.use_bottleneck = use_bottleneck
        self.apply(weight_init)

        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = max(latent_dim // 2, 8)
            self.bottleneck = InformationBottleneck(latent_dim, bottleneck_dim, norm, drop)
    
    def encode_vae(self, x: torch.Tensor):
        """VAE encoding: returns (mu, var)"""
        h = self.encoder_backbone(x)
        mu = self.mu_head(h)
        var = torch.nn.functional.softplus(self.var_head(h)) + self.var_eps
        return mu, var
    
    def encode_ae(self, x: torch.Tensor):
        """AE encoding: returns latent directly"""
        return self.encoder(x)
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Standard forward pass."""
        if self.use_vae:
            mu, var = self.encode_vae(x)
            z = self.reparameterize(mu, var)
        else:
            z = self.encode_ae(x)
            mu, var = None, None
        
        if self.use_bottleneck:
            z_le, z_ld = self.bottleneck(z)
            x_hat = self.decoder(z)
            x_le_hat = self.decoder(z_ld)
            return x_hat, z, x_le_hat, z_le, mu, var
        else:
            x_hat = self.decoder(z)
            return x_hat, z, None, None, mu, var

class DPMMODEModel(BaseModel):
    """DPMMODE: AE/VAE with DPMM clustering."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_dims: Optional[list] = None,
        decoder_dims: Optional[list] = None,
        norm_type: str = "bn",
        dropout_rate: float = 0.1,
        dpmm_warmup_ratio: float = 0.6,
        dpmm_loss_weight: float = 1.0,
        dpmm_refit_interval: int = 10,
        n_components: int = 50,
        dpmm_loss_type: Literal['nll', 'kl', 'energy', 'student_t', 'mmd', 'soft_nll'] = 'kl',
        student_t_df: float = 3.0,
        mmd_bandwidth: float = 1.0,
        model_name: str = "DPMMODE",
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        use_vae: bool = False,
        kl_weight: float = 0.1,
        **kwargs):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=encoder_dims or [], model_name=model_name)
        self.ae = DPMMAutoEncoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims or [256, 128],
            latent_dim=latent_dim,
            decoder_dims=decoder_dims or [128, 256],
            norm=norm_type,
            drop=dropout_rate,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            use_vae=use_vae)
        self.dpmm_warmup_ratio = dpmm_warmup_ratio
        self.dpmm_loss_weight = dpmm_loss_weight
        self.dpmm_refit_interval = dpmm_refit_interval
        self.n_components = n_components
        self.dpmm_loss_type = dpmm_loss_type
        self.student_t_df = student_t_df
        self.mmd_bandwidth = mmd_bandwidth
        self.dpmm_params = None
        self.dpmm_fitted = False
        self.recon_loss_fn = nn.MSELoss()
        self.use_vae = use_vae
        self.kl_weight = kl_weight

        # Safety guard: VAE Gaussian prior N(0,I) conflicts with DPMM clustering prior
        if self.use_vae and self.dpmm_loss_weight > 0:
            import warnings
            warnings.warn(
                f"{self.model_name}: use_vae=True with dpmm_loss_weight={self.dpmm_loss_weight} "
                f"creates conflicting KL objectives (Gaussian N(0,I) vs DPMM mixture). "
                f"Forcing dpmm_loss_weight=0.0 and dpmm_warmup_ratio=1.0.",
                UserWarning, stacklevel=2)
            self.dpmm_loss_weight = 0.0
            self.dpmm_warmup_ratio = 1.0

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space."""
        if self.ae.use_vae:
            mu, var = self.ae.encode_vae(x)
            return mu
        else:
            return self.ae.encode_ae(x)

    def extract_latent(self, data_loader, device='cuda',
                       return_reconstructions: bool = False, **kwargs):
        """Extract latent representations from DataLoader."""
        self.eval()
        self.to(device)
        
        latents, recons = [], []
        
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                z = self.encode(x, **batch_kwargs)
                latents.append(z.cpu().numpy())
                
                if return_reconstructions:
                    recons.append(self.decode(z).cpu().numpy())
        
        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode from latent space"""
        return self.ae.decoder(z)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass (Phase 1: AE/VAE training without ODE)"""
        if self.ae.use_bottleneck:
            x_hat, z, x_le_hat, z_ld, mu, var = self.ae(x)
            recon = (x_hat, x_le_hat)
        else:
            x_hat, z, _, _, mu, var = self.ae(x)
            recon = (x_hat,)
        result = {"reconstruction": recon, "latent": z}
        if mu is not None:
            result["mu"] = mu
            result["var"] = var
        return result

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss: reconstruction + DPMM regularization (Phase 1 only)
        """
        loss_dict = {}
        
        if self.ae.use_bottleneck:
            x_hat, x_le_hat = outputs["reconstruction"]
            recon_ae = self.recon_loss_fn(x_hat, x)
            recon_bottleneck = self.recon_loss_fn(x_le_hat, x)
            recon = (recon_ae + recon_bottleneck) / 2.0
            
            loss_dict["recon_ae"] = recon_ae
            loss_dict["recon_bottleneck"] = recon_bottleneck
        else:
            x_hat, = outputs["reconstruction"]
            recon = self.recon_loss_fn(x_hat, x)
            loss_dict["recon_ae"] = recon
        
        loss_dict["recon_loss"] = recon
        
        # DPMM clustering loss
        dpmm = torch.tensor(0.0, device=x.device)
        if self.dpmm_fitted and self.dpmm_params is not None:
            if self.dpmm_loss_type == 'nll':
                dpmm = self._dpmm_loss_nll(outputs["latent"])
            elif self.dpmm_loss_type == 'kl':
                dpmm = self._dpmm_loss_kl(outputs["latent"])
            elif self.dpmm_loss_type == 'energy':
                dpmm = self._dpmm_loss_energy(outputs["latent"])
            elif self.dpmm_loss_type == 'student_t':
                dpmm = self._dpmm_loss_student_t(outputs["latent"])
            elif self.dpmm_loss_type == 'mmd':
                dpmm = self._dpmm_loss_mmd(outputs["latent"])
            elif self.dpmm_loss_type == 'soft_nll':
                dpmm = self._dpmm_loss_soft_nll(outputs["latent"])
        
        loss_dict["dpmm_loss"] = dpmm
        
        # VAE KL divergence loss (if using VAE mode)
        kl_vae = torch.tensor(0.0, device=x.device)
        if self.use_vae and "mu" in outputs and "var" in outputs:
            kl_vae = self._kl_gaussian(outputs["mu"], outputs["var"])
            loss_dict["kl_vae"] = kl_vae
        
        # Compute total weighted loss
        total = recon + self.dpmm_loss_weight * dpmm + self.kl_weight * kl_vae
        loss_dict["total_loss"] = total
        
        return loss_dict

    def _update_dpmm_params(self, bgm: BayesianGaussianMixture, device: torch.device):
        """Extract DPMM parameters from fitted sklearn model"""
        means = torch.as_tensor(bgm.means_, dtype=torch.float32, device=device)
        weight_concentration = torch.as_tensor(bgm.weight_concentration_, dtype=torch.float32, device=device)
        precisions_cholesky = torch.as_tensor(bgm.precisions_cholesky_, dtype=torch.float32, device=device)
        weights = torch.as_tensor(bgm.weights_, dtype=torch.float32, device=device)
        
        # Numerical stability
        precisions_cholesky = torch.clamp(precisions_cholesky, min=1e-6, max=1e6)
        
        self.dpmm_params = {
            "means": means,
            "weight_concentration": weight_concentration,
            "precisions_cholesky": precisions_cholesky,
            "weights": weights,
            "covariances": 1.0 / (precisions_cholesky ** 2 + 1e-10),  # diagonal covariance
        }
        self.dpmm_fitted = True

    def _dpmm_loss_nll(self, z: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood formulation using stick-breaking weights
        """
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        # Compute log weights using stick-breaking construction
        digamma_sum = torch.special.digamma(dp["weight_concentration"][0] + dp["weight_concentration"][1])
        digamma_a = torch.special.digamma(dp["weight_concentration"][0])
        digamma_b = torch.special.digamma(dp["weight_concentration"][1])

        log_weights_b = torch.cat(
            [torch.zeros(1, device=z.device), torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1]]
        )
        log_weights = digamma_a - digamma_sum + log_weights_b

        # Compute log determinant
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        # Mahalanobis distance
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)

        # Log Gaussian
        log_gauss = -0.5 * (n_features * math.log(2.0 * math.pi) + mahalanobis) + log_det
        log_likelihood = torch.logsumexp(log_gauss + log_weights, dim=1)
        
        # Negative log-likelihood (clamped to ensure non-negative)
        nll = -log_likelihood
        return torch.clamp(nll.mean(), min=0.0)

    def _dpmm_loss_kl(self, z: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood under DPMM (always non-negative)
        """
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        # Get mixture weights
        weights = dp["weights"]
        log_weights = torch.log(weights + eps)

        # Compute log determinant
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        # Mahalanobis distance
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)  # (batch, n_components, dim)
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)

        # Log probability per component
        log_gauss = -0.5 * (n_features * math.log(2.0 * math.pi) + mahalanobis) + log_det
        
        # Weighted log probability (mixture model)
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)
        
        # Negative log-likelihood (always non-negative)
        nll = -log_prob
        
        # Clamp to ensure numerical stability (should already be non-negative)
        return torch.clamp(nll.mean(), min=0.0)

    def _dpmm_loss_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Energy-based distance to nearest cluster
        """
        dp = self.dpmm_params
        weights = dp["weights"]
        
        # Compute weighted distances to all clusters
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)  # (batch, n_components, dim)
        
        # Use precision as inverse variance (higher precision = tighter cluster)
        precisions = dp["precisions_cholesky"] ** 2
        weighted_dist = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)  # (batch, n_components)
        
        # Soft assignment: use Gaussian kernel
        temperature = 1.0
        soft_assign = torch.softmax(-weighted_dist / temperature, dim=1)  # (batch, n_components)
        
        # Weighted energy
        energy = torch.sum(soft_assign * weighted_dist, dim=1).mean()
        
        return energy

    def _dpmm_loss_student_t(self, z: torch.Tensor) -> torch.Tensor:
        """
        Student's t-distribution instead of Gaussian
        """
        dp = self.dpmm_params
        n_features = z.size(1)
        df = self.student_t_df  # degrees of freedom
        eps = 1e-10

        # Get mixture weights
        weights = dp["weights"]
        log_weights = torch.log(weights + eps)

        # Compute Mahalanobis distance
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        precisions = dp["precisions_cholesky"] ** 2
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)

        # Student's t log probability
        # log p(x) = log(Gamma((df+d)/2)) - log(Gamma(df/2)) - (d/2)log(df*pi) 
        #            + 0.5*log(det(Precision)) - ((df+d)/2)*log(1 + mahalanobis/df)
        
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        
        log_student_t = (
            torch.lgamma(torch.tensor((df + n_features) / 2.0, device=z.device))
            - torch.lgamma(torch.tensor(df / 2.0, device=z.device))
            - (n_features / 2.0) * math.log(df * math.pi)
            + log_det
            - ((df + n_features) / 2.0) * torch.log(1.0 + mahalanobis / df)
        )
        
        # Mixture log probability
        log_prob = torch.logsumexp(log_student_t + log_weights, dim=1)
        
        # Negative log-likelihood (clamped instead of abs to avoid gradient inversion)
        nll = -log_prob
        return torch.clamp(nll.mean(), min=0.0)

    def _dpmm_loss_mmd(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maximum Mean Discrepancy between latent samples and DPMM samples
        """
        dp = self.dpmm_params
        
        # Sample from DPMM
        n_samples = z.size(0)
        
        # Sample component indices based on weights
        component_samples = torch.multinomial(dp["weights"], n_samples, replacement=True)
        
        # Sample from selected components
        means = dp["means"][component_samples]  # (n_samples, dim)
        stds = torch.sqrt(dp["covariances"][component_samples])  # (n_samples, dim)
        dpmm_samples = means + stds * torch.randn_like(means)
        
        # Compute MMD with RBF kernel
        def rbf_kernel(x, y, bandwidth):
            # x: (n, d), y: (m, d)
            xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (n, 1)
            yy = torch.sum(y ** 2, dim=1, keepdim=True)  # (m, 1)
            xy = torch.mm(x, y.T)  # (n, m)
            dists = xx + yy.T - 2 * xy  # (n, m)
            return torch.exp(-dists / (2 * bandwidth ** 2))
        
        bandwidth = self.mmd_bandwidth
        
        k_xx = rbf_kernel(z, z, bandwidth).mean()
        k_yy = rbf_kernel(dpmm_samples, dpmm_samples, bandwidth).mean()
        k_xy = rbf_kernel(z, dpmm_samples, bandwidth).mean()
        
        mmd = k_xx - 2 * k_xy + k_yy
        
        # MMD is always non-negative, but numerical errors can make it slightly negative
        return torch.relu(mmd)

    def _dpmm_loss_soft_nll(self, z: torch.Tensor) -> torch.Tensor:
        """
        Softened NLL using smooth approximation
        """
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        # Get mixture weights
        weights = dp["weights"]
        log_weights = torch.log(weights + eps)

        # Compute log determinant
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        # Mahalanobis distance
        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff ** 2 * precisions.unsqueeze(0), dim=2)

        # Log Gaussian
        log_gauss = -0.5 * (n_features * math.log(2.0 * math.pi) + mahalanobis) + log_det
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)
        
        # Convert to probability (normalized)
        # Use a temperature to prevent saturation
        temperature = 2.0
        prob = torch.exp(log_prob / temperature)
        
        # Brier score style: (1 - p)^2
        loss = ((1.0 - prob) ** 2).mean()
        
        return loss
    
    def _kl_gaussian(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between N(mu, var) and N(0, I)
        """
        # KL divergence formula for Gaussian
        kl_per_sample = 0.5 * torch.sum(var + mu ** 2 - 1.0 - torch.log(var + 1e-10), dim=1)
        return kl_per_sample.mean()

    def _refit_dpmm(self, train_loader, device: torch.device, verbose: int = 1) -> bool:
        """Refit DPMM on current latent representations"""
        self.eval()
        z_all = []
        
        with torch.no_grad():
            for batch in train_loader:
                x, _ = self._prepare_batch(batch, device)
                z = self.encode(x)
                z_all.append(z.cpu().numpy())
        
        z_all = np.concatenate(z_all, axis=0)
        z_all = z_all + np.random.normal(0, 1e-6, z_all.shape)
        
        try:
            bgm = BayesianGaussianMixture(
                n_components=self.n_components,
                weight_concentration_prior=1.0,
                mean_precision_prior=0.1,
                covariance_type="diag",
                init_params="kmeans",
                max_iter=100,
                tol=1e-3,
                random_state=42)
            bgm.fit(z_all)
            
            if not np.isfinite(bgm.lower_bound_):
                if verbose >= 1:
                    print("Warning: DPMM fitting resulted in non-finite lower bound")
                return False
            
            self._update_dpmm_params(bgm, device=device)
            return True
            
        except Exception as e:
            if verbose >= 1:
                print(f"Warning: DPMM fitting failed with error: {e}")
            return False

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        lr: float = 1e-4,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 20,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 1e-5,
        **kwargs):
        """
        Phase 1: Train AE/VAE with periodic DPMM refitting
        
        Args:
            weight_decay: AdamW weight decay (L2 regularization).
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.total_epochs = epochs
        self.current_epoch = 0
        
        dpmm_warmup_epochs = int(epochs * self.dpmm_warmup_ratio)
        best_loss = float('inf')
        patience_counter = 0
        
        # Initialize loss tracking
        train_losses = []
        recon_losses = []
        dpmm_losses = []
        kl_vae_losses = []
        recon_ae_losses = []
        recon_bottleneck_losses = []
        flow_losses = []
        has_flow_loss = False

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.current_epoch = epoch
            if epoch < dpmm_warmup_epochs:
                self.dpmm_fitted = False
            elif epoch == dpmm_warmup_epochs or (epoch - dpmm_warmup_epochs) % self.dpmm_refit_interval == 0:
                do_print_refit = (verbose >= 1) and (
                    ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
                )
                if do_print_refit:
                    print(f"Epoch {epoch+1}: Refitting DPMM...")
                success = self._refit_dpmm(train_loader, torch.device(device), verbose=verbose)
                if not success and do_print_refit:
                    print(f"Epoch {epoch+1}: DPMM refitting failed")

            self.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_dpmm = 0.0
            epoch_kl_vae = 0.0
            epoch_recon_ae = 0.0
            epoch_recon_bottleneck = 0.0
            epoch_flow = 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]
                
                if not torch.isfinite(loss):
                    if verbose >= 2:
                        print(f"Warning: Non-finite loss at epoch {epoch+1}, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                
                # Accumulate loss components
                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_dpmm += loss_dict["dpmm_loss"].item()
                
                if "kl_vae" in loss_dict:
                    epoch_kl_vae += loss_dict["kl_vae"].item()
                
                if "recon_ae" in loss_dict:
                    epoch_recon_ae += loss_dict["recon_ae"].item()
                if "recon_bottleneck" in loss_dict:
                    epoch_recon_bottleneck += loss_dict["recon_bottleneck"].item()
                if "flow_loss" in loss_dict:
                    epoch_flow += loss_dict["flow_loss"].item()
                    has_flow_loss = True
                
                n_batches += 1

            if n_batches == 0:
                continue

            # Compute averages
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_dpmm = epoch_dpmm / n_batches
            avg_kl_vae = epoch_kl_vae / n_batches if self.use_vae else 0.0
            avg_recon_ae = epoch_recon_ae / n_batches
            avg_recon_bottleneck = epoch_recon_bottleneck / n_batches if self.ae.use_bottleneck else 0.0
            avg_flow = epoch_flow / n_batches if has_flow_loss else 0.0
            
            # Record losses
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            dpmm_losses.append(avg_dpmm)
            if self.use_vae:
                kl_vae_losses.append(avg_kl_vae)
            recon_ae_losses.append(avg_recon_ae)
            if self.ae.use_bottleneck:
                recon_bottleneck_losses.append(avg_recon_bottleneck)
            if has_flow_loss:
                flow_losses.append(avg_flow)

            # Verbose logging
            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if do_print:
                phase = "Warmup" if epoch < dpmm_warmup_epochs else f"DPMM[{self.dpmm_loss_type}]"
                log_msg = (f"Epoch {epoch+1:3d}/{epochs} [Phase1-{phase}] | "
                          f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | DPMM: {avg_dpmm:.4f}")
                
                if self.use_vae:
                    log_msg += f" | KL: {avg_kl_vae:.4f}"
                
                if self.ae.use_bottleneck:
                    log_msg += f" | AE: {avg_recon_ae:.4f} | BN: {avg_recon_bottleneck:.4f}"
                if has_flow_loss:
                    log_msg += f" | FM: {avg_flow:.4f}"
                
                print(log_msg)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Return training history
        result = {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
            "dpmm_loss": dpmm_losses,
            "recon_ae": recon_ae_losses,
        }
        
        if self.use_vae:
            result["kl_vae"] = kl_vae_losses
        
        if self.ae.use_bottleneck:
            result["recon_bottleneck"] = recon_bottleneck_losses
        if has_flow_loss:
            result["flow_loss"] = flow_losses
        
        return result


def create_dpmmode_model(input_dim: int, latent_dim: int = 32, **kwargs) -> DPMMODEModel:
    """
    Create DPMMODE model.
    
    Args:
        input_dim: Number of input features (genes)
        latent_dim: Latent space dimension (16-128)
    
    Critical Parameters:
        encoder_dims/decoder_dims: Network dimensions (default: [256,128]/[128,256])
        n_components: DPMM components (20-100)
        dpmm_loss_type: 'nll', 'kl', 'energy', 'student_t', 'mmd', 'soft_nll'
        dpmm_warmup_ratio: DPMM warmup fraction (0.3-0.8)
        use_bottleneck: Enable information bottleneck
        use_vae: Use VAE instead of AE
    """
    # Strip ODE-related kwargs for backward compatibility
    for _k in ("use_latent_dynamics", "ae_reg", "ode_reg", "ode_epochs",
               "ode_lr", "ode_consistency_weight", "ode_recon_weight", "use_ode"):
        kwargs.pop(_k, None)
    return DPMMODEModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
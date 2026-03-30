"""
统一模型接口 - 导入所有模型

约定：
- 每个模型提供统一接口（BaseModel 子类）
- 每个模型提供 create_* 工厂函数
"""
from .base_model import BaseModel
from .cellblast_model import CellBLASTModel, create_cellblast_model
from .clear_model import CLEARModel, create_clear_model
from .disentanglement_vae_model import DisentanglementVAEModel, create_disentanglement_vae_model
from .scalex_model import SCALEXModel, create_scalex_model
from .scdac_model import create_scdac_model, scDACModel
from .scdeepcluster_model import create_scdeepcluster_model, scDeepClusterModel
from .scdhmap_model import create_scdhmap_model, scDHMapModel
from .scdiffusion_model import create_scdiffusion_model, scDiffusionModel
from .scgcc_model import create_scgcc_model, scGCCModel
from .scgnn_model import create_scgnn_model, scGNNModel
from .scsmd_model import create_scsmd_model, scSMDModel
from .sivae_model import create_sivae_model, siVAEModel

# GM-VAE (requires geoopt)
try:
    from .gmvae_model import GMVAEModel, create_gmvae_model
    _GEOOPT_AVAILABLE = True
except ImportError:
    _GEOOPT_AVAILABLE = False

# scVI-family (requires scvi-tools)
try:
    from .scvi_family_model import (
        PeakVIModel,
        PoissonVIModel,
        SCVIModel,
        create_peakvi_model,
        create_poissonvi_model,
        create_scvi_model,
    )
    _SCVI_AVAILABLE = True
except ImportError:
    _SCVI_AVAILABLE = False

__all__ = [
    # base
    "BaseModel",
    # models
    "CellBLASTModel",
    "SCALEXModel",
    "scDiffusionModel",
    "siVAEModel",
    "CLEARModel",
    "scDACModel",
    "scDeepClusterModel",
    "scDHMapModel",
    "scGNNModel",
    "scGCCModel",
    "scSMDModel",
    "DisentanglementVAEModel",
    # factories
    "create_cellblast_model",
    "create_scalex_model",
    "create_scdiffusion_model",
    "create_sivae_model",
    "create_clear_model",
    "create_scdac_model",
    "create_scdeepcluster_model",
    "create_scdhmap_model",
    "create_scgnn_model",
    "create_scgcc_model",
    "create_scsmd_model",
    "create_disentanglement_vae_model",
]

if _GEOOPT_AVAILABLE:
    __all__ += ["GMVAEModel", "create_gmvae_model"]

if _SCVI_AVAILABLE:
    __all__ += [
        "SCVIModel", "PeakVIModel", "PoissonVIModel",
        "create_scvi_model", "create_peakvi_model", "create_poissonvi_model",
    ]

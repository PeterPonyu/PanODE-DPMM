"""
Euclidean Normal Distribution
Standard Gaussian distribution in Euclidean space
"""

from .distribution import Distribution
from .layers import VanillaDecoderLayer, VanillaEncoderLayer
from .prior import get_prior

__all__ = [
    'Distribution',
    'VanillaEncoderLayer',
    'VanillaDecoderLayer',
    'get_prior',
]

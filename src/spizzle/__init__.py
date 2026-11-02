from importlib.metadata import version, PackageNotFoundError

from .geometry import corners, jacobian, overlap_triplets
from .operator import build_operator, drizzle_operator, Operator, Coadd
from .covariance import Covariance, NoiseFactor
from .pixmap import pixel_map
from .interpolation import blot


try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "dev"


__all__ = [
    '__version__',
    'corners',
    'jacobian',
    'overlap_triplets',
    'build_operator',
    'drizzle_operator',
    'Operator',
    'Coadd',
    'Covariance',
    'NoiseFactor',
    'pixel_map',
    'blot',
]

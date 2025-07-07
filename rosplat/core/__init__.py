"""
This file makes the 'core' directory a Python package and exposes key components.
"""

from .gaussian_representation import GaussianData
from . import util

__all__ = ['GaussianData', 'util']

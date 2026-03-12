"""
pixel3dmm module - Self-contained pixel3dmm FLAME initializer for ehm-tracker.

Provides optimization-based FLAME parameter estimation from single face images,
using pixel3dmm's DINO ViT + transformer network for UV/normal prediction
followed by iterative FLAME fitting.
"""
from .initializer import Pixel3DMMInitializer

__all__ = ['Pixel3DMMInitializer']

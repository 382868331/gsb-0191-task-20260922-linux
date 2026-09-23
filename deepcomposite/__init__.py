"""Reusable two-layer deep-sample transparent compositing module."""

from .core import (
    MAX_ITEMS,
    CompositeResult,
    DeepCompositeError,
    LedgerEntry,
    Surface,
    VolumeSegment,
    composite,
)

__all__ = [
    "MAX_ITEMS",
    "CompositeResult",
    "DeepCompositeError",
    "LedgerEntry",
    "Surface",
    "VolumeSegment",
    "composite",
]

__version__ = "1.0.0"

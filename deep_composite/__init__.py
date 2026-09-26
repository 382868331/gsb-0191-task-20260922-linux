"""双层深度样本透明合成模块(deep_composite)。

公开接口:
    Surface, VolumeSegment   —— 输入条目(构造时即校验)
    composite(deep_samples, *, z_far=None) -> CompositeResult
    apply_background(result, background) -> (r, g, b)
    CompositeResult(color, transmittance, ledger)
    LedgerEntry
    MAX_ITEMS
    DeepCompositeError, ValidationError, NegativeSigmaError, ItemLimitError
"""

from .core import (
    MAX_ITEMS,
    CompositeResult,
    DeepCompositeError,
    ItemLimitError,
    LedgerEntry,
    NegativeSigmaError,
    SimplificationResult,
    Surface,
    ValidationError,
    VolumeSegment,
    apply_background,
    composite,
    simplify_surfaces,
)

__all__ = [
    "MAX_ITEMS",
    "CompositeResult",
    "DeepCompositeError",
    "ItemLimitError",
    "LedgerEntry",
    "NegativeSigmaError",
    "SimplificationResult",
    "Surface",
    "ValidationError",
    "VolumeSegment",
    "apply_background",
    "composite",
    "simplify_surfaces",
]

__version__ = "1.0.0"

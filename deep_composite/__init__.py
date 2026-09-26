"""双层深度样本透明合成模块(deep_composite)。

公开接口:
    Surface, VolumeSegment   —— 输入条目(构造时即校验)
    composite(deep_samples, *, z_far=None) -> CompositeResult
    apply_background(result, background) -> (r, g, b)
    simplify_surfaces(surfaces, *, tolerance) -> SimplifyResult
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
    Surface,
    ValidationError,
    VolumeSegment,
    apply_background,
    composite,
    SimplifyResult,
    simplify_surfaces,
)

__all__ = [
    "MAX_ITEMS",
    "CompositeResult",
    "DeepCompositeError",
    "ItemLimitError",
    "LedgerEntry",
    "NegativeSigmaError",
    "Surface",
    "ValidationError",
    "VolumeSegment",
    "apply_background",
    "composite",
    "SimplifyResult",
    "simplify_surfaces",
]

__version__ = "1.1.0"

"""双层深度样本透明合成模块(deep_composite)。

公开接口:
    Surface, VolumeSegment   —— 输入条目(构造时即校验)
    composite(deep_samples, *, z_far=None) -> CompositeResult
    composite_range(deep_samples, z_near, z_far) -> CompositeResult
    combine_results(first, second) -> CompositeResult
    apply_background(result, background) -> (r, g, b)
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
    combine_results,
    composite,
    composite_range,
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
    "combine_results",
    "composite",
    "composite_range",
]

__version__ = "1.0.0"

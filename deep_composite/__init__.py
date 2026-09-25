"""双层深度样本透明合成模块(deep_composite)。

公开接口:
    Surface, VolumeSegment   —— 输入条目(构造时即校验)
    composite(deep_samples, *, z_far=None) -> CompositeResult
    composite_range(deep_samples, z_near, z_far) -> CompositeResult
        半开深度窗 [z_near, z_far) 的独立合成(窗前遮挡不带入)
    combine_results(first, second) -> CompositeResult
        按 C = C1 + T1*C2, T = T1*T2 拼回相邻窗口
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

__version__ = "1.1.0"

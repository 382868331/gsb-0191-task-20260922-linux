"""双层深度样本的透明合成核心算法。

坐标约定:
    z 轴沿视线方向单调递增(离相机越远 z 越大)。所有样本按 z 从前到后
    (front-to-back) 依次合成。

预乘约定:
    - 表面携带预乘 RGBA: (r, g, b, a), 其中 rgb 已是 "颜色 * alpha"。
      四个通道均须在 [0, 1] 内。本模块不强制 rgb <= a(超预乘数据不拒绝),
      但输入被假定为预乘含义, 合成时直接使用, 不再乘 alpha。
    - 体段携带发射系数 emission=(er, eg, eb)(每单位长度, 非负, 可大于 1)
      与消光系数 sigma(每单位长度, 非负; 标量表示灰度/无色消光,
      长度 3 的序列 (sigma_r, sigma_g, sigma_b) 表示有色消光)。
    - 输出 color 为预乘 RGB(相对透明背景), transmittance 为逐通道透过率 RGB;
      叠到背景 bg 上: out = color + transmittance * bg。

合成规则:
    1. 所有体段端点把深度轴切成不交叠小区间; 每个小区间内重叠体段的
       sigma 相加、emission 相加, 按均匀介质解析积分:
           C_k = e_k / sigma_k * (1 - exp(-sigma_k * L))   (sigma_k > 0)
           C_k = e_k * L                                    (sigma_k == 0)
           T_k = exp(-sigma_k * L)
    2. 同一 z 处: 先处理"在该 z 结束的体段区间", 再按 ID 升序合成该 z 的
       所有表面, 之后才轮到"在该 z 开始的体段区间"。
    3. 表面按 (z, ID) 排序; ID 为 int(拒绝 bool)或 str, 混合类型时
       int 排在 str 前(确定性 tie-break, 不报错)。
    4. 可选 z_far 截断远平面: z > z_far 的表面被丢弃, 体段被截到 z_far。

错误语义:
    所有非法输入在 composite() 返回前抛出, 函数为纯函数, 失败不留部分
    结果。错误类型见模块顶部异常类; 消息包含条目序号/ID 与字段名, 可定位。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

__all__ = [
    "MAX_ITEMS",
    "DeepCompositeError",
    "ValidationError",
    "NegativeSigmaError",
    "ItemLimitError",
    "Surface",
    "VolumeSegment",
    "LedgerEntry",
    "CompositeResult",
    "composite",
    "apply_background",
]

#: 每次 composite() 调用允许的最大条目数(接口边界)。
MAX_ITEMS = 100

Rgb = Tuple[float, float, float]
ItemId = Union[int, str]

#: sigma 允许标量(灰度消光)或逐通道三元组(有色消光)。
Sigma = Union[float, Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class DeepCompositeError(Exception):
    """本模块所有错误的基类。"""


class ValidationError(DeepCompositeError, ValueError):
    """输入非法(类型、NaN/Inf、越界、负发射、z1<z0 等)。"""


class NegativeSigmaError(ValidationError):
    """sigma 为负。单独成类以便调用方精确捕获。"""


class ItemLimitError(ValidationError):
    """单次调用的条目数超过 MAX_ITEMS。"""


# ---------------------------------------------------------------------------
# 输入校验辅助
# ---------------------------------------------------------------------------

def _check_real(value, name: str, ctx: str) -> float:
    """实数校验: 拒绝 bool、非数值、NaN、Infinity。返回 float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{ctx}: 字段 {name} 须为实数, 得到 {type(value).__name__}({value!r})"
        )
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValidationError(f"{ctx}: 字段 {name} 须为有限值, 得到 {v!r}")
    return v


def _check_id(value, ctx: str) -> ItemId:
    """ID 校验: int(拒绝 bool)或 str。"""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValidationError(
            f"{ctx}: 字段 id 须为 int 或 str, 得到 {type(value).__name__}({value!r})"
        )
    return value


def _id_sort_key(item_id: ItemId):
    # int 与 str 混合时确定性排序: int 在前。
    return (0, item_id) if isinstance(item_id, int) else (1, item_id)


def _check_rgb_triple(value, name: str, ctx: str, allow_above_one: bool) -> Rgb:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            f"{ctx}: 字段 {name} 须为长度为 3 的数值序列, 得到 {type(value).__name__}"
        )
    if len(value) != 3:
        raise ValidationError(
            f"{ctx}: 字段 {name} 长度须为 3, 得到 {len(value)}"
        )
    out = []
    for i, comp in enumerate(value):
        v = _check_real(comp, f"{name}[{i}]", ctx)
        if v < 0.0:
            raise ValidationError(f"{ctx}: 字段 {name}[{i}] 须非负, 得到 {v!r}")
        if not allow_above_one and v > 1.0:
            raise ValidationError(
                f"{ctx}: 字段 {name}[{i}] 须在 [0, 1] 内, 得到 {v!r}"
            )
        out.append(v)
    return (out[0], out[1], out[2])


def _check_sigma(value, ctx: str) -> Rgb:
    """消光系数校验: 接受非负标量(广播为三通道)或长度 3 的非负通道序列。

    返回归一化后的 (sigma_r, sigma_g, sigma_b)。拒绝 bool、NaN/Inf、负值
    与长度错误的序列; 不静默纠正非法输入。
    """
    if isinstance(value, bool):
        raise ValidationError(
            f"{ctx}: 字段 sigma 不能为 bool, 得到 {value!r}"
        )
    if isinstance(value, (int, float)):
        v = _check_real(value, "sigma", ctx)
        if v < 0.0:
            raise NegativeSigmaError(f"{ctx}: sigma 须非负, 得到 {v!r}")
        return (v, v, v)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            f"{ctx}: 字段 sigma 须为非负标量或长度为 3 的数值序列, "
            f"得到 {type(value).__name__}"
        )
    if len(value) != 3:
        raise ValidationError(
            f"{ctx}: 字段 sigma 为序列时长度须为 3, 得到 {len(value)}"
        )
    channels = []
    for i, comp in enumerate(value):
        v = _check_real(comp, f"sigma[{i}]", ctx)
        if v < 0.0:
            raise NegativeSigmaError(
                f"{ctx}: 字段 sigma[{i}] 须非负, 得到 {v!r}"
            )
        channels.append(v)
    return (channels[0], channels[1], channels[2])


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Surface:
    """深度 z 处的半透明表面, rgba 为预乘 (r, g, b, a), 各通道在 [0, 1]。"""

    z: float
    rgba: Tuple[float, float, float, float]
    id: ItemId = 0

    def __post_init__(self):
        ctx = f"Surface(id={self.id!r})"
        object.__setattr__(self, "id", _check_id(self.id, ctx))
        object.__setattr__(self, "z", _check_real(self.z, "z", ctx))
        rgba = self.rgba
        if isinstance(rgba, (str, bytes)) or not isinstance(rgba, Sequence) or len(rgba) != 4:
            raise ValidationError(
                f"{ctx}: 字段 rgba 须为长度为 4 的数值序列"
            )
        object.__setattr__(
            self, "rgba", tuple(_check_real(c, f"rgba[{i}]", ctx) for i, c in enumerate(rgba))
        )
        for i, c in enumerate(self.rgba):
            if not 0.0 <= c <= 1.0:
                raise ValidationError(
                    f"{ctx}: 字段 rgba[{i}] 须在 [0, 1] 内(预乘约定), 得到 {c!r}"
                )


@dataclass(frozen=True)
class VolumeSegment:
    """均匀体段 [z0, z1]: emission 为每单位长度发射系数(RGB, 非负),
    sigma 为每单位长度消光系数, 接受标量(灰度, 三通道相同)或逐通道三元组
    (sigma_r, sigma_g, sigma_b, 各分量非负)。z0 == z1 为零长段(无贡献)。
    """

    z0: float
    z1: float
    emission: Tuple[float, float, float]
    sigma: Sigma
    id: ItemId = 0

    def __post_init__(self):
        ctx = f"VolumeSegment(id={self.id!r})"
        object.__setattr__(self, "id", _check_id(self.id, ctx))
        z0 = _check_real(self.z0, "z0", ctx)
        z1 = _check_real(self.z1, "z1", ctx)
        if z1 < z0:
            raise ValidationError(f"{ctx}: 须满足 z1 >= z0, 得到 z0={z0!r}, z1={z1!r}")
        object.__setattr__(self, "z0", z0)
        object.__setattr__(self, "z1", z1)
        object.__setattr__(
            self, "emission", _check_rgb_triple(self.emission, "emission", ctx, allow_above_one=True)
        )
        object.__setattr__(self, "sigma", _check_sigma(self.sigma, ctx))


@dataclass(frozen=True)
class LedgerEntry:
    """贡献账本条目: 一个被合成的元素(合并后的体段小区间或单个表面)。

    contribution 为该元素对最终色的贡献(已乘以前方透过率);
    全部条目的 contribution 求和即 CompositeResult.color(可加性)。
    """

    kind: str  # "volume" 或 "surface"
    ids: Tuple[ItemId, ...]  # 参与该元素的原始条目 ID(体段小区间可能合并多个)
    z0: float
    z1: float  # 表面时 z1 == z0
    contribution: Rgb
    transmittance_before: Rgb
    transmittance_after: Rgb


@dataclass(frozen=True)
class CompositeResult:
    """合成结果: 预乘最终色、逐通道透过率 RGB、贡献账本。"""

    color: Rgb
    transmittance: Rgb
    ledger: Tuple[LedgerEntry, ...]


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def _segment_color_and_transmittance(emission: Rgb, sigma: Rgb, length: float):
    """均匀小区间的逐通道解析积分。返回 (C_rgb, T_rgb)。

    每个颜色通道独立积分: 某通道 sigma_k == 0(或零长)时该通道取
    C_k = e_k * L, T_k = 1 —— 不做除法, 发射贡献完整保留。
    """
    color = [0.0, 0.0, 0.0]
    trans = [0.0, 0.0, 0.0]
    for k in range(3):
        x = sigma[k] * length
        if x == 0.0:
            color[k] = emission[k] * length
            trans[k] = 1.0
        else:
            # (1 - exp(-x)) / x 用 expm1 保持小 x 精度
            color[k] = emission[k] * length * (-math.expm1(-x) / x)
            trans[k] = math.exp(-x)
    return (color[0], color[1], color[2]), (trans[0], trans[1], trans[2])


def composite(
    deep_samples: Iterable[Union[Surface, VolumeSegment]],
    *,
    z_far: Optional[float] = None,
) -> CompositeResult:
    """合成一个像素的深度样本, 返回最终色、透过率与贡献账本。

    参数:
        deep_samples: Surface / VolumeSegment 的可迭代对象, 至多 MAX_ITEMS 项。
        z_far: 可选截断远平面; z > z_far 的表面被丢弃, 体段截断到 z_far。

    抛出:
        ItemLimitError: 条目数超过 MAX_ITEMS。
        NegativeSigmaError: 某体段 sigma 为负(构造时已校验, 此处为防御)。
        ValidationError: 其他非法输入(类型、NaN/Inf、越界等)。

    本函数为纯函数: 任何校验失败都在产生结果之前抛出, 不留部分变更。
    """
    if isinstance(deep_samples, (str, bytes)):
        raise ValidationError("composite: deep_samples 须为样本条目的可迭代对象")
    try:
        items = list(deep_samples)
    except TypeError as exc:
        raise ValidationError(
            f"composite: deep_samples 不可迭代: {exc}"
        ) from exc

    if len(items) > MAX_ITEMS:
        raise ItemLimitError(
            f"composite: 条目数 {len(items)} 超过上限 MAX_ITEMS={MAX_ITEMS}"
        )

    for i, it in enumerate(items):
        if not isinstance(it, (Surface, VolumeSegment)):
            raise ValidationError(
                f"composite: 第 {i} 项须为 Surface 或 VolumeSegment, "
                f"得到 {type(it).__name__}({it!r})"
            )

    if z_far is not None:
        z_far = _check_real(z_far, "z_far", "composite")

    # --- 截断远平面(显式变换, 不静默: 被丢弃/截断的语义写入文档) ---
    surfaces = []
    volumes = []
    for it in items:
        if isinstance(it, Surface):
            if z_far is None or it.z <= z_far:
                surfaces.append(it)
        else:
            if it.z0 == it.z1:
                continue  # 零长段无贡献
            z1 = it.z1 if z_far is None else min(it.z1, z_far)
            if z1 <= it.z0:
                continue  # 完全在远平面之外
            volumes.append((it.z0, z1, it.emission, it.sigma, it.id))

    # --- 体段端点切分: 收集所有边界坐标 ---
    coords = set()
    for z0, z1, _e, _s, _i in volumes:
        coords.add(z0)
        coords.add(z1)
    for s in surfaces:
        coords.add(s.z)
    ordered = sorted(coords)

    # --- 从前到后合成 ---
    acc = [0.0, 0.0, 0.0]
    transmittance = [1.0, 1.0, 1.0]
    ledger = []

    # 每个 z 坐标上的表面(按 ID 排序)
    surfaces_at = {}
    for s in surfaces:
        surfaces_at.setdefault(s.z, []).append(s)
    for z in surfaces_at:
        surfaces_at[z].sort(key=lambda s: _id_sort_key(s.id))

    for idx, z in enumerate(ordered):
        # 1) 先合成 (prev_z, z) 区间 —— 即"在 z 结束的体段部分"
        if idx > 0:
            a, b = ordered[idx - 1], z
            if b > a:
                sig = [0.0, 0.0, 0.0]
                emi = [0.0, 0.0, 0.0]
                ids = []
                for z0, z1, e, s, iid in volumes:
                    if z0 <= a and z1 >= b:
                        sig[0] += s[0]
                        sig[1] += s[1]
                        sig[2] += s[2]
                        emi[0] += e[0]
                        emi[1] += e[1]
                        emi[2] += e[2]
                        ids.append(iid)
                if ids:
                    c_seg, t_seg = _segment_color_and_transmittance(
                        (emi[0], emi[1], emi[2]), (sig[0], sig[1], sig[2]), b - a
                    )
                    t_before = (transmittance[0], transmittance[1], transmittance[2])
                    acc[0] += transmittance[0] * c_seg[0]
                    acc[1] += transmittance[1] * c_seg[1]
                    acc[2] += transmittance[2] * c_seg[2]
                    transmittance[0] *= t_seg[0]
                    transmittance[1] *= t_seg[1]
                    transmittance[2] *= t_seg[2]
                    ledger.append(
                        LedgerEntry(
                            kind="volume",
                            ids=tuple(ids),
                            z0=a,
                            z1=b,
                            contribution=(
                                t_before[0] * c_seg[0],
                                t_before[1] * c_seg[1],
                                t_before[2] * c_seg[2],
                            ),
                            transmittance_before=t_before,
                            transmittance_after=(
                                transmittance[0], transmittance[1], transmittance[2]
                            ),
                        )
                    )
        # 2) 再合成 z 处的表面(按 ID 升序)—— 之后才是从 z 开始的体段
        for s in surfaces_at.get(z, ()):
            r, g, b_, alpha = s.rgba
            t_before = (transmittance[0], transmittance[1], transmittance[2])
            acc[0] += transmittance[0] * r
            acc[1] += transmittance[1] * g
            acc[2] += transmittance[2] * b_
            residual = 1.0 - alpha
            transmittance[0] *= residual
            transmittance[1] *= residual
            transmittance[2] *= residual
            ledger.append(
                LedgerEntry(
                    kind="surface",
                    ids=(s.id,),
                    z0=z,
                    z1=z,
                    contribution=(
                        t_before[0] * r,
                        t_before[1] * g,
                        t_before[2] * b_,
                    ),
                    transmittance_before=t_before,
                    transmittance_after=(
                        transmittance[0], transmittance[1], transmittance[2]
                    ),
                )
            )

    return CompositeResult(
        color=(acc[0], acc[1], acc[2]),
        transmittance=(transmittance[0], transmittance[1], transmittance[2]),
        ledger=tuple(ledger),
    )


def apply_background(result: CompositeResult, background: Sequence[float]) -> Rgb:
    """把合成结果叠到背景色上: out_k = color_k + transmittance_k * background_k。"""
    bg = _check_rgb_triple(background, "background", "apply_background", allow_above_one=True)
    t = _check_rgb_triple(result.transmittance, "transmittance", "apply_background", allow_above_one=True)
    return (
        result.color[0] + t[0] * bg[0],
        result.color[1] + t[1] * bg[1],
        result.color[2] + t[2] * bg[2],
    )

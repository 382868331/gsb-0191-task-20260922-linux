"""双层深度样本的透明合成核心算法。

坐标约定:
    z 轴沿视线方向单调递增(离相机越远 z 越大)。所有样本按 z 从前到后
    (front-to-back) 依次合成。

预乘约定:
    - 表面携带预乘 RGBA: (r, g, b, a), 其中 rgb 已是 "颜色 * alpha"。
      四个通道均须在 [0, 1] 内。本模块不强制 rgb <= a(超预乘数据不拒绝),
      但输入被假定为预乘含义, 合成时直接使用, 不再乘 alpha。
    - 体段携带发射系数 emission=(er, eg, eb)(每单位长度, 非负, 可大于 1)
      与消光系数 sigma(每单位长度, 标量, 非负; 消光为无色/灰度)。
    - 输出 color 为预乘 RGB(相对透明背景), transmittance 为整束透过率;
      叠到背景 bg 上: out = color + transmittance * bg。

合成规则:
    1. 所有体段端点把深度轴切成不交叠小区间; 每个小区间内重叠体段的
       sigma 相加、emission 相加, 按均匀介质解析积分:
           C = e / sigma * (1 - exp(-sigma * L))   (sigma > 0)
           C = e * L                                (sigma == 0)
           T = exp(-sigma * L)
    2. 同一 z 处: 先处理"在该 z 结束的体段区间", 再按 ID 升序合成该 z 的
       所有表面, 之后才轮到"在该 z 开始的体段区间"。
    3. 表面按 (z, ID) 排序; ID 为 int(拒绝 bool)或 str, 混合类型时
       int 排在 str 前(确定性 tie-break, 不报错)。
    4. 可选 z_far 截断远平面: z > z_far 的表面被丢弃, 体段被截到 z_far。

深度窗 composite_range(z_near, z_far):
    半开窗 [z_near, z_far): 表面满足 z_near <= z < z_far 才计入
    (恰在 near 的表面计入, 恰在 far 的表面排除); 体段裁剪到窗内,
    跨边界时按截后长度重新积分。窗口结果是"独立窗口"结果: 窗内透过率
    从 1 起算, 窗前遮挡不带入(窗外样本即使传入也只做裁剪, 不产生贡献)。
    相邻窗口可按 C = C1 + T1*C2, T = T1*T2 拼回(见 combine_results)。
    空窗(含 z_near == z_far)返回零色与 T=1; z_far < z_near 拒绝。

错误语义:
    所有非法输入在 composite()/composite_range() 返回前抛出, 函数为纯
    函数, 失败不留部分结果。错误类型见模块顶部异常类; 消息包含条目
    序号/ID 与字段名, 可定位。
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
    "composite_range",
    "combine_results",
    "apply_background",
]

#: 每次 composite()/composite_range() 调用允许的最大条目数(接口边界)。
MAX_ITEMS = 100

Rgb = Tuple[float, float, float]
ItemId = Union[int, str]


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
    sigma 为每单位长度消光系数(标量, 非负)。z0 == z1 为零长段(无贡献)。
    """

    z0: float
    z1: float
    emission: Tuple[float, float, float]
    sigma: float
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
        sigma = _check_real(self.sigma, "sigma", ctx)
        if sigma < 0.0:
            raise NegativeSigmaError(f"{ctx}: sigma 须非负, 得到 {sigma!r}")
        object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True)
class LedgerEntry:
    """贡献账本条目: 一个被合成的元素(合并后的体段小区间或单个表面)。

    contribution 为该元素对最终色的贡献(已乘以前方透过率);
    全部条目的 contribution 求和即 CompositeResult.color(可加性)。

    对 composite_range 产生的窗口结果, z0/z1 为体段裁剪到窗内后的区间,
    ids 为原始样本 ID; transmittance_before/after 是窗口内局部透过率
    (从 1 起算), 不含窗前遮挡。
    """

    kind: str  # "volume" 或 "surface"
    ids: Tuple[ItemId, ...]  # 参与该元素的原始条目 ID(体段小区间可能合并多个)
    z0: float
    z1: float  # 表面时 z1 == z0
    contribution: Rgb
    transmittance_before: float
    transmittance_after: float


@dataclass(frozen=True)
class CompositeResult:
    """合成结果: 预乘最终色、整束透过率、贡献账本。"""

    color: Rgb
    transmittance: float
    ledger: Tuple[LedgerEntry, ...]


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def _segment_color_and_transmittance(emission: Rgb, sigma: float, length: float):
    """均匀小区间的解析积分。返回 (C_rgb, T)。"""
    x = sigma * length
    if x == 0.0:
        # sigma == 0(或零长): C = e * L, T = 1
        return (emission[0] * length, emission[1] * length, emission[2] * length), 1.0
    # (1 - exp(-x)) / x 用 expm1 保持小 x 精度
    factor = -math.expm1(-x) / x  # 无量纲
    t = math.exp(-x)
    return (
        emission[0] * length * factor,
        emission[1] * length * factor,
        emission[2] * length * factor,
    ), t


def _prepare_items(
    deep_samples: Iterable[Union[Surface, VolumeSegment]], ctx: str
) -> Tuple[list, list]:
    """物化并校验输入条目(上限、类型), 返回 (surfaces, volumes 原始列表)。"""
    if isinstance(deep_samples, (str, bytes)):
        raise ValidationError(f"{ctx}: deep_samples 须为样本条目的可迭代对象")
    try:
        items = list(deep_samples)
    except TypeError as exc:
        raise ValidationError(f"{ctx}: deep_samples 不可迭代: {exc}") from exc

    if len(items) > MAX_ITEMS:
        raise ItemLimitError(
            f"{ctx}: 条目数 {len(items)} 超过上限 MAX_ITEMS={MAX_ITEMS}"
        )

    surfaces = []
    volumes = []
    for i, it in enumerate(items):
        if isinstance(it, Surface):
            surfaces.append(it)
        elif isinstance(it, VolumeSegment):
            volumes.append(it)
        else:
            raise ValidationError(
                f"{ctx}: 第 {i} 项须为 Surface 或 VolumeSegment, "
                f"得到 {type(it).__name__}({it!r})"
            )
    return surfaces, volumes


def _run_composite(
    surfaces: Sequence[Surface],
    volumes: Sequence[Tuple[float, float, Rgb, float, ItemId]],
) -> CompositeResult:
    """按端点切分并 front-to-back 合成。

    volumes 为已裁剪的 (z0, z1, emission, sigma, id) 五元组(零长段已剔除);
    surfaces 为已按窗口筛选的表面。透过率从 1 起算, 与窗前内容无关。
    """
    # --- 体段端点 + 表面深度: 收集所有边界坐标 ---
    coords = set()
    for z0, z1, _e, _s, _i in volumes:
        coords.add(z0)
        coords.add(z1)
    for s in surfaces:
        coords.add(s.z)
    ordered = sorted(coords)

    # 每个 z 坐标上的表面(按 ID 排序)
    surfaces_at = {}
    for s in surfaces:
        surfaces_at.setdefault(s.z, []).append(s)
    for z in surfaces_at:
        surfaces_at[z].sort(key=lambda s: _id_sort_key(s.id))

    # --- 从前到后合成 ---
    acc = [0.0, 0.0, 0.0]
    transmittance = 1.0
    ledger = []

    for idx, z in enumerate(ordered):
        # 1) 先合成 (prev_z, z) 区间 —— 即"在该 z 结束的体段部分"
        if idx > 0:
            a, b = ordered[idx - 1], z
            if b > a:
                sig = 0.0
                emi = [0.0, 0.0, 0.0]
                ids = []
                for z0, z1, e, s, iid in volumes:
                    if z0 <= a and z1 >= b:
                        sig += s
                        emi[0] += e[0]
                        emi[1] += e[1]
                        emi[2] += e[2]
                        ids.append(iid)
                if ids:
                    c_seg, t_seg = _segment_color_and_transmittance(
                        (emi[0], emi[1], emi[2]), sig, b - a
                    )
                    t_before = transmittance
                    acc[0] += transmittance * c_seg[0]
                    acc[1] += transmittance * c_seg[1]
                    acc[2] += transmittance * c_seg[2]
                    transmittance *= t_seg
                    ledger.append(
                        LedgerEntry(
                            kind="volume",
                            ids=tuple(ids),
                            z0=a,
                            z1=b,
                            contribution=(
                                t_before * c_seg[0],
                                t_before * c_seg[1],
                                t_before * c_seg[2],
                            ),
                            transmittance_before=t_before,
                            transmittance_after=transmittance,
                        )
                    )
        # 2) 再合成 z 处的表面(按 ID 升序)—— 之后才是从 z 开始的体段
        for s in surfaces_at.get(z, ()):
            r, g, b_, alpha = s.rgba
            t_before = transmittance
            acc[0] += transmittance * r
            acc[1] += transmittance * g
            acc[2] += transmittance * b_
            transmittance *= 1.0 - alpha
            ledger.append(
                LedgerEntry(
                    kind="surface",
                    ids=(s.id,),
                    z0=z,
                    z1=z,
                    contribution=(t_before * r, t_before * g, t_before * b_),
                    transmittance_before=t_before,
                    transmittance_after=transmittance,
                )
            )

    return CompositeResult(
        color=(acc[0], acc[1], acc[2]),
        transmittance=transmittance,
        ledger=tuple(ledger),
    )


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
    surfaces_raw, volumes_raw = _prepare_items(deep_samples, "composite")

    if z_far is not None:
        z_far = _check_real(z_far, "z_far", "composite")

    # --- 截断远平面(显式变换, 不静默: 被丢弃/截断的语义写入文档) ---
    surfaces = []
    volumes = []
    for it in surfaces_raw:
        if z_far is None or it.z <= z_far:
            surfaces.append(it)
    for it in volumes_raw:
        if it.z0 == it.z1:
            continue  # 零长段无贡献
        z1 = it.z1 if z_far is None else min(it.z1, z_far)
        if z1 <= it.z0:
            continue  # 完全在远平面之外
        volumes.append((it.z0, z1, it.emission, it.sigma, it.id))

    return _run_composite(surfaces, volumes)


def composite_range(
    deep_samples: Iterable[Union[Surface, VolumeSegment]],
    z_near: float,
    z_far: float,
) -> CompositeResult:
    """只合成半开深度窗 [z_near, z_far) 内的内容, 返回该窗的独立结果。

    参数:
        deep_samples: Surface / VolumeSegment 的可迭代对象, 至多 MAX_ITEMS 项。
        z_near: 窗的近端(闭端), 须为有限实数。
        z_far: 窗的远端(开端), 须为有限实数且 z_far >= z_near。

    窗口语义(半开 [z_near, z_far)):
        - 表面: z_near <= z < z_far 才计入; 恰在 z_near 的表面计入,
          恰在 z_far 的表面排除。
        - 体段: 先与窗口求交裁剪 (max(z0,z_near), min(z1,z_far));
          跨边界的体段按截后长度重新解析积分(不沿用截前长度),
          账本记录裁剪后的 [z0, z1] 与原样本 ID。
        - 结果为"独立窗口"结果: 窗内透过率从 1 起算, 窗前遮挡不带入。
          传入窗前样本不会(也不能)影响窗口颜色; 需要还原整段遮挡时,
          由调用方按 C = C1 + T1*C2、T = T1*T2 拼回(combine_results)。

    边界情形:
        - 空窗(无窗内表面与体段), 以及 z_near == z_far: color=零、
          transmittance=1、ledger 为空。
        - z_far < z_near(反向窗)抛 ValidationError。

    抛出:
        ItemLimitError / ValidationError: 同 composite()。
    """
    surfaces_raw, volumes_raw = _prepare_items(deep_samples, "composite_range")
    z_near = _check_real(z_near, "z_near", "composite_range")
    z_far = _check_real(z_far, "z_far", "composite_range")
    if z_far < z_near:
        raise ValidationError(
            f"composite_range: 须满足 z_far >= z_near, "
            f"得到 z_near={z_near!r}, z_far={z_far!r}(反向窗口)"
        )

    # --- 半开窗筛选表面: [z_near, z_far) ---
    surfaces = [s for s in surfaces_raw if z_near <= s.z < z_far]

    # --- 体段与窗口求交; 窗外段与零长截段剔除 ---
    volumes = []
    for it in volumes_raw:
        z0 = max(it.z0, z_near)
        z1 = min(it.z1, z_far)
        if z1 <= z0:
            continue  # 窗外、零长, 或仅端点相贴(半开约定下无窗内长度)
        volumes.append((z0, z1, it.emission, it.sigma, it.id))

    return _run_composite(surfaces, volumes)


def combine_results(
    first: CompositeResult, second: CompositeResult
) -> CompositeResult:
    """把沿视线相邻(不重叠)的两个窗口结果从前到后拼回。

    设 first 为近端窗口 (C1, T1), second 为远端窗口 (C2, T2), 则:
        C = C1 + T1 * C2
        T = T1 * T2
    两个输入窗口本身仍是各自独立的结果(本函数不改写它们)。拼回结果的
    账本保持全局一致: 后段条目的 contribution 乘以 T1、局部透过率
    transmittance_before/after 也乘以 T1(等价于把窗内局部透过率换算为
    全局透过率), 原始样本 ID 与裁剪范围 [z0, z1] 原样保留。这样拼回
    账本在接缝处链条连续(前段末条 after=T1, 后段首条 before=T1), 且
    全部条目 contribution 之和恰等于拼回颜色(可加性精确保留)。
    拼回颜色与直接按 C1 + T1*C2 计算仅差双精度舍入(约 1e-15)。
    """
    if not isinstance(first, CompositeResult) or not isinstance(second, CompositeResult):
        raise ValidationError(
            "combine_results: 两个参数均须为 CompositeResult, "
            f"得到 {type(first).__name__} 与 {type(second).__name__}"
        )
    t1 = first.transmittance

    # 后段账本全局化: 颜色贡献与局部透过率统一乘以前窗透过率 T1。
    rescaled_second = []
    for e in second.ledger:
        rescaled_second.append(
            LedgerEntry(
                kind=e.kind,
                ids=e.ids,
                z0=e.z0,
                z1=e.z1,
                contribution=(
                    t1 * e.contribution[0],
                    t1 * e.contribution[1],
                    t1 * e.contribution[2],
                ),
                transmittance_before=t1 * e.transmittance_before,
                transmittance_after=t1 * e.transmittance_after,
            )
        )
    ledger = tuple(first.ledger) + tuple(rescaled_second)

    # 颜色直接由账本贡献求和得到, 保证"账本求和恰等于颜色"精确成立;
    # 数学上即 C = C1 + T1*C2。
    acc = [0.0, 0.0, 0.0]
    for e in ledger:
        acc[0] += e.contribution[0]
        acc[1] += e.contribution[1]
        acc[2] += e.contribution[2]

    return CompositeResult(
        color=(acc[0], acc[1], acc[2]),
        transmittance=t1 * second.transmittance,
        ledger=ledger,
    )


def apply_background(result: CompositeResult, background: Sequence[float]) -> Rgb:
    """把合成结果叠到背景色上: out = color + transmittance * background。"""
    bg = _check_rgb_triple(background, "background", "apply_background", allow_above_one=True)
    t = result.transmittance
    return (
        result.color[0] + t * bg[0],
        result.color[1] + t * bg[1],
        result.color[2] + t * bg[2],
    )

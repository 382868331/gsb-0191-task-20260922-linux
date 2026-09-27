# deep_composite — 双层深度样本的透明合成

离线渲染用的可复用合成模块: 把一个像素的一组深度样本(半透明表面 + 均匀体段)
按从前到后(front-to-back)顺序解析合成为最终色、透过率与贡献账本。

- **Python 版本**: 3.14.7(仅标准库, 安装 Python 后无需任何额外依赖)
- **平台**: Linux 原生; 核心、测试、演示全部离线
- **运行测试**: `python -m unittest discover -s tests -v`
- **运行演示**: `python demo.py`(固定输入, 真实计算, 约毫秒级完成)

## 接口

```python
from deep_composite import (
    Surface, VolumeSegment, composite, apply_background,
    CompositeResult, LedgerEntry, MAX_ITEMS,
    DeepCompositeError, ValidationError, NegativeSigmaError, ItemLimitError,
)

result = composite(deep_samples, *, z_far=None)
```

### 输入条目(构造时即校验)

| 类型 | 字段 | 含义 |
|---|---|---|
| `Surface(z, rgba, id=0)` | `z` 有限实数; `rgba` 为**预乘** `(r,g,b,a)`, 四通道均在 `[0,1]` | 深度 z 处的半透明表面 |
| `VolumeSegment(z0, z1, emission, sigma, id=0)` | `z1 >= z0`; `emission` 为每单位长度发射系数 RGB(非负有限, 可大于 1); `sigma` 为每单位长度消光系数, 接受**非负标量**(灰度消光, 三通道相同)或**长度 3 的非负通道序列** `(σr, σg, σb)`(有色消光) | 均匀体段 |

- `id`: `int`(**拒绝 `bool`**)或 `str`, 用于同深度 tie-break; int 与 str 混合时 int 排前。
- `z_far`: 可选截断远平面。`z > z_far` 的表面被丢弃, 体段被截断到 `z_far`(完全在外的丢弃)。

### 输出

`CompositeResult(color, transmittance, ledger)`:

- `color`: 预乘 RGB 最终色(相对透明背景)。叠到背景上: `apply_background(result, bg)` = `color + transmittance * bg`。
- `transmittance`: **逐通道透过率** `(Tr, Tg, Tb)`, 各分量 ∈ [0,1]; 有色消光下三个通道可以不同。
- `ledger`: 贡献账本, 每个被合成元素一条 `LedgerEntry(kind, ids, z0, z1, contribution, transmittance_before, transmittance_after)`, 其中透过率字段均为逐通道 RGB 三元组。**全部条目 `contribution` 求和恰等于 `color`**(可加性, demo 与测试均核验)。

## 预乘与合成约定

- 表面 RGBA 为预乘: rgb 已含 alpha, 合成时直接使用, 不再乘 alpha; 不强制 `rgb <= a`(超预乘数据不拒绝, 但按预乘语义解释)。
- 体段端点把深度轴切成不交叠小区间; 每个小区间内重叠体段的 **sigma 逐通道相加、emission 逐通道相加**, 每个颜色通道 k 独立按均匀介质解析积分(深度顺序、表面与体段的相对先后对三通道保持一致):
  - `Ck = ek/σk · (1 − exp(−σk·L))`,`Tk = exp(−σk·L)`;某通道 `σk = 0` 时该通道取 `Ck = ek·L, Tk = 1`。
  - 零吸收通道不做除法, 发光贡献完整保留, 不会出现 0/0 或丢光。
  - 小 `σk·L` 用 `expm1` 稳定求值, 避免精度损失。
- 标量 `sigma` 在构造时归一化为 `(σ, σ, σ)`; 因此标量灰度消光与等参数三元组给出**逐位相同**的结果(测试 `test_triple_equal_to_scalar_grayscale` 核验)。
- 同一 z 处的顺序: **先结束于 z 的体段区间 → z 处的表面(按 ID 升序)→ 从 z 开始的体段区间**。
- 表面整体按 `(z, id)` 排序, 全部元素 front-to-back 逐通道合成: `Ck += Tk·ck; Tk *= (1−a)`(表面, α 对三通道相同)或 `Tk *= T_seg,k`(体段)。

## 输入上限与错误语义

- **上限**: 每次 `composite()` 调用至多 `MAX_ITEMS = 100` 项, 超出抛 `ItemLimitError`。
- 所有数值输入拒绝 `NaN`/`Infinity`; 整型语义参数(id)拒绝 `bool`; 数值字段同样拒绝 `bool`。
- 标量 `sigma < 0`, 或三元组任一通道 `< 0`, 抛 `NegativeSigmaError`; 负发射、rgba 越界、`z1 < z0`、sigma 序列长度错误、错误类型等抛 `ValidationError`。
- 错误消息包含条目上下文(id)与字段名, 可定位; 不静默纠正任何非法输入。
- `composite()` 为纯函数: 校验失败在产生结果之前抛出, 不留部分变更; `z0 == z1` 的零长体段是合法 no-op。

## 误差口径

- 体段积分为闭式解析解, 非数值积分; 主要误差来源是 `exp`/`expm1` 与浮点累加(双精度)。
- 对量级正常的输入(σ·L ≲ 数百), 结果相对误差在 **1e-12** 量级; 测试对解析参考使用 `1e-12` 容差, 对极小 σ 的级数极限使用 `1e-9`。
- 分段不变性(一段切多段)在 **1e-12** 内成立; 账本可加性为**精确相等**(贡献与累加用同一组浮点值)。

## 设计取舍与已知限制

- **消光支持逐通道(RGB)**: `sigma` 传标量即灰度(三通道相同), 传三元组即有色吸收; 发射本来就是 RGB。
- **重叠体段在小区间内合并**(σ、e 分别相加)后再积分, 账本按合并后的小区间记账(`ids` 列出全部参与条目), 不做逐条目的重叠区归因。
- 仅实现本题契约的合成模型: 不含散射、折射、非均匀介质或体纹理; 不宣称兼容任何行业深阴影图/deep pixel 标准。
- 表面不强制 `rgb <= a` 的预乘不变式(仅要求各通道 ∈ [0,1])。
- 复杂度 O(N²) 量级(N ≤ 100, 每区间扫描活动体段), 对接口上限足够; 非大数据结构优化目标。

## 文件结构

```
deep_composite/      # 可导入库(核心算法, 无第三方依赖)
  __init__.py        # 公开接口导出
  core.py            # 数据结构、校验、合成算法
tests/test_composite.py  # 单元测试(含逐通道有色消光解析参考, 独立手算)
demo.py              # 固定输入演示: 正常结果 + 账本可加核验 + 负 sigma 拒绝
```

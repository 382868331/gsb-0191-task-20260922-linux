# deep_composite — 双层深度样本的透明合成

离线渲染用的可复用合成模块: 把一个像素的一组深度样本(半透明表面 + 均匀体段)
按从前到后(front-to-back)顺序解析合成为最终色、透过率与贡献账本。

- **Python 版本**: 3.14.7(仅标准库, 安装 Python 后无需任何额外依赖)
- **平台**: Linux 原生; 核心、测试、演示全部离线
- **运行测试**: `python -m unittest discover -s tests -v`(45 个用例)
- **运行演示**: `python demo.py`(固定输入, 真实计算, 约毫秒级完成; 含正常结果、边界与实际错误)

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
| `VolumeSegment(z0, z1, emission, sigma, id=0)` | `z1 >= z0`; `emission` 为每单位长度发射系数 RGB(非负有限, 可大于 1); `sigma` 为每单位长度消光系数(标量, 非负有限) | 均匀体段 |

- `id`: `int`(**拒绝 `bool`**)或 `str`, 用于同深度 tie-break; int 与 str 混合时 int 排前。
- `z_far`: 可选截断远平面。`z > z_far` 的表面被丢弃, 体段被截断到 `z_far`(完全在外的丢弃)。

### 输出

`CompositeResult(color, transmittance, ledger)`:

- `color`: 预乘 RGB 最终色(相对透明背景)。叠到背景上: `apply_background(result, bg)` = `color + transmittance * bg`。
- `transmittance`: 整束透过率 T ∈ [0,1]。
- `ledger`: 贡献账本, 每个被合成元素一条 `LedgerEntry(kind, ids, z0, z1, contribution, transmittance_before, transmittance_after)`。**全部条目 `contribution` 求和恰等于 `color`**(可加性, demo 与测试均核验)。账本穷举全部小区间与表面: 被不透明表面遮挡的后方元素仍有条目, 但其 `transmittance_before` 与 `contribution` 为 `0`。

## 预乘与合成约定

- 表面 RGBA 为预乘: rgb 已含 alpha, 合成时直接使用, 不再乘 alpha; 不强制 `rgb <= a`(超预乘数据不拒绝, 但按预乘语义解释)。
- 体段端点把深度轴切成不交叠小区间; 每个小区间内重叠体段的 **sigma 相加、emission 相加**, 按均匀介质解析积分:
  - `C = e/σ · (1 − exp(−σ·L))`,`T = exp(−σ·L)`;`σ = 0` 时 `C = e·L, T = 1`。
  - 小 `σ·L` 用 `expm1` 稳定求值, 避免 0/0 精度损失。
- 同一 z 处的顺序: **先结束于 z 的体段区间 → z 处的表面(按 ID 升序)→ 从 z 开始的体段区间**。
- 表面整体按 `(z, id)` 排序, 全部元素 front-to-back 合成: `C += T·c; T *= (1−a)`(表面)或 `T *= T_seg`(体段)。

## 输入上限与错误语义

- **上限**: 每次 `composite()` 调用至多 `MAX_ITEMS = 100` 项, 超出抛 `ItemLimitError`。
- 所有数值输入拒绝 `NaN`/`Infinity`; 整型语义参数(id)拒绝 `bool`; 数值字段同样拒绝 `bool`。
- `sigma < 0` 抛 `NegativeSigmaError`; 负发射、rgba 越界、`z1 < z0`、错误类型等抛 `ValidationError`。
- 错误消息包含条目上下文(id)与字段名, 可定位; 不静默纠正任何非法输入。
- `composite()` 为纯函数: 校验失败在产生结果之前抛出, 不留部分变更; `z0 == z1` 的零长体段是合法 no-op。

## 误差口径

- 体段积分为闭式解析解, 非数值积分; 主要误差来源是 `exp`/`expm1` 与浮点累加(双精度)。
- 对量级正常的输入(σ·L ≲ 数百), 结果相对误差在 **1e-12** 量级; 测试对解析参考使用 `1e-12` 容差, 对极小 σ 的级数极限使用 `1e-9`。
- 分段不变性(一段切多段)在 **1e-12** 内成立; 账本可加性为**精确相等**(贡献与累加用同一组浮点值)。

## 设计取舍与已知限制

- **消光为标量(灰度)**: `sigma` 不随波长变化, 不支持有色消光; 发射为 RGB。
- **重叠体段在小区间内合并**(σ、e 分别相加)后再积分, 账本按合并后的小区间记账(`ids` 列出全部参与条目), 不做逐条目的重叠区归因。
- 仅实现本题契约的合成模型: 不含散射、折射、非均匀介质或体纹理; 不宣称兼容任何行业深阴影图/deep pixel 标准。
- 表面不强制 `rgb <= a` 的预乘不变式(仅要求各通道 ∈ [0,1])。
- 复杂度 O(N²) 量级(N ≤ 100, 每区间扫描活动体段), 对接口上限足够; 非大数据结构优化目标。

## 文件结构

```
deep_composite/      # 可导入库(核心算法, 无第三方依赖)
  __init__.py        # 公开接口导出
  core.py            # 数据结构、校验、合成算法
tests/test_composite.py  # 45 个单元测试(解析参考独立手算 + 60 场景枚举核对)
demo.py              # 固定输入演示: 正常结果(独立解析核对) + 零消光/遮挡边界 + 实际错误
```

## 测试组织

- 单段/双重叠/嵌套/三方重叠的解析参考(测试内独立用 `math.exp` 手算, 不调用被测核心)。
- 边界: 零消光(单段与重叠)、零长体段、极厚强消光体段后的数值全黑、表面恰在 `z_far`。
- 遮挡: 不透明表面遮挡后方表面与体段、半透明表面减半后方贡献、体段消光衰减后方表面、不透明表面与同 z 开始体段的顺序。
- `TestEnumeratedReference`: 固定种子的小规模枚举(60 个随机场景), 与一份独立的逐区间(slab)参考实现逐通道核对, 同时核验账本可加与透过率链; 枚举仅用于交叉验证, 期望值不来自被测实现。
- 失败用例: 负 sigma、NaN/Inf、bool、rgba/emission 形状与类型、错误 id 类型、`z_far` 非法、条目数超限、错误条目类型、生成器输入、纯函数性(重复调用一致、输入不被修改)。

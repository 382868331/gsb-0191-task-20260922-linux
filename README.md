# deep_composite — 双层深度样本的透明合成

离线渲染用的可复用合成模块: 把一个像素的一组深度样本(半透明表面 + 均匀体段)
按从前到后(front-to-back)顺序解析合成为最终色、透过率与贡献账本。

- **Python 版本**: 3.14.7(仅标准库, 安装 Python 后无需任何额外依赖)
- **平台**: Linux 原生; 核心、测试、演示全部离线
- **运行测试**: `./.venv/bin/python -m unittest discover -s tests -v`
- **运行演示**: `./.venv/bin/python demo.py`(固定输入, 真实计算, 约毫秒级完成)

## 接口

```python
from deep_composite import (
    Surface, VolumeSegment, composite, composite_range, combine_results,
    apply_background,
    CompositeResult, LedgerEntry, MAX_ITEMS,
    DeepCompositeError, ValidationError, NegativeSigmaError, ItemLimitError,
)

result = composite(deep_samples, *, z_far=None)          # 全量合成
window = composite_range(deep_samples, z_near, z_far)    # 半开深度窗 [z_near, z_far)
joined = combine_results(window1, window2)               # C=C1+T1*C2, T=T1*T2
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
- `ledger`: 贡献账本, 每个被合成元素一条 `LedgerEntry(kind, ids, z0, z1, contribution, transmittance_before, transmittance_after)`。**全部条目 `contribution` 求和恰等于 `color`**(可加性, demo 与测试均核验)。

### 深度窗 `composite_range(samples, z_near, z_far)`

只合成**半开窗口 `[z_near, z_far)`** 内的内容, 返回窗口自身的颜色与透过率, 供调用方考察某段深度对最终画面的独立贡献:

- **独立性(关键语义)**: 窗口结果与窗前内容无关, 透过率从 **1.0** 起算。窗前即使有不透明表面, 也不会遮挡窗口, 窗口结果也不包含任何窗前贡献 —— 它不是"从相机看到的窗内画面"。需要从相机所见时, 用 `combine_results` 把前面的窗口依次拼上。
- **体段**: 先裁剪到窗口, 再**按截后长度重新解析积分**(不携带窗外的光学厚度或发射)。跨边界体段在窗内的部分与普通体段完全同式。
- **表面**: 恰在 `z_near` **计入**, 恰在 `z_far` **排除**(半开)。
- **空窗口**: 窗内无表面且无正长度体段交集(含 `z_near == z_far`)时, 返回零色 `(0,0,0)`、透过率 `1`、空账本。
- **反向窗口**: `z_far < z_near`(或 NaN/Infinity 边界)抛 `ValidationError`。
- **账本**: 保留**原始样本 ID** 与**裁剪后范围**(跨边界体段条目的 `z0/z1` 为截后坐标)。
- `z_far` 参数注意: `composite(..., z_far=x)` 保持旧契约(仅远向截断, 表面恰在 x **保留**); 半开语义只属于 `composite_range`。

### 窗口拼回 `combine_results(r1, r2)`

沿深度轴相邻、不交的两个窗口, 按 front-to-back 拼接:

- `C = C1 + T1·C2`,`T = T1·T2`;多窗依次 fold 即可。任意切点的分窗拼回与 `composite()` 全量结果一致(测试以 1e-12 容差核验, 且全量结果另有独立解析手算参考)。
- 拼回后的账本为两段账本的顺序拼接, 第二段条目的贡献与透过率统一乘 `T1`, 账本链仍连续、贡献求和仍恰等于拼回色。
- 本函数只做代数拼接, 不校验两窗口坐标是否真的相邻(调用方负责)。

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
- `composite_range` 的 `z_near/z_far` 须为有限实数且 `z_far >= z_near`, 否则抛 `ValidationError`; 条目上限与类型校验同样适用。
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
- `combine_results` 不检查窗口是否真的相邻/不交, 重叠或乱序拼接在代数上仍可计算, 但不保证等于全量合成。
- 深度窗不引入新的体采样模型: 窗内仍用与 `composite` 相同的均匀介质解析积分, 只改变参与积分的深度范围。

## 文件结构

```
deep_composite/      # 可导入库(核心算法, 无第三方依赖)
  __init__.py        # 公开接口导出
  core.py            # 数据结构、校验、合成算法(composite / composite_range / combine_results)
tests/
  test_composite.py       # 26 个旧回归单元测试(解析参考独立手算)
  test_composite_range.py # 19 个深度窗测试(手算边界、独立解析全量参考、分窗拼回)
demo.py              # 固定输入演示: 正常结果 + 账本可加 + 深度窗拼回 + 拒绝边界
```

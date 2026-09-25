# deep_composite — 双层深度样本的透明合成

离线渲染用的可复用合成模块: 把一个像素的一组深度样本(半透明表面 + 均匀体段)
按从前到后(front-to-back)顺序解析合成为最终色、透过率与贡献账本。
除整段合成外, 支持只取某个**半开深度窗** `[z_near, z_far)` 的独立贡献,
并可把任意相邻窗口按渲染方程拼回整段。

- **Python 版本**: 3.14.7(仅标准库, 安装 Python 后无需任何额外依赖)
- **平台**: Linux 原生; 核心、测试、演示全部离线, 不使用 Docker、外部服务或账号
- **运行测试**: `./.venv/bin/python -m unittest discover -s tests -v`
- **运行演示**: `./.venv/bin/python demo.py`(固定输入, 真实计算, 约毫秒级完成, 远低于 90 秒)

## 接口

```python
from deep_composite import (
    Surface, VolumeSegment, composite, composite_range, combine_results,
    apply_background, CompositeResult, LedgerEntry, MAX_ITEMS,
    DeepCompositeError, ValidationError, NegativeSigmaError, ItemLimitError,
)

result = composite(deep_samples, *, z_far=None)                 # 整段合成
window = composite_range(deep_samples, z_near, z_far)           # 半开窗 [near, far) 独立合成
joined = combine_results(window_near, window_far)               # C=C1+T1*C2, T=T1*T2
```

### 输入条目(构造时即校验)

| 类型 | 字段 | 含义 |
|---|---|---|
| `Surface(z, rgba, id=0)` | `z` 有限实数; `rgba` 为**预乘** `(r,g,b,a)`, 四通道均在 `[0,1]` | 深度 z 处的半透明表面 |
| `VolumeSegment(z0, z1, emission, sigma, id=0)` | `z1 >= z0`; `emission` 为每单位长度发射系数 RGB(非负有限, 可大于 1); `sigma` 为每单位长度消光系数(标量, 非负有限) | 均匀体段 |

- `id`: `int`(**拒绝 `bool`**)或 `str`, 用于同深度 tie-break; int 与 str 混合时 int 排前。
- `z_far`(仅 `composite`): 可选截断远平面。`z > z_far` 的表面被丢弃, 体段被截断到 `z_far`(完全在外的丢弃)。

### 输出

`CompositeResult(color, transmittance, ledger)`:

- `color`: 预乘 RGB 最终色(相对透明背景)。叠到背景上: `apply_background(result, bg)` = `color + transmittance * bg`。
- `transmittance`: 整束透过率 T ∈ [0,1]。
- `ledger`: 贡献账本, 每个被合成元素一条 `LedgerEntry(kind, ids, z0, z1, contribution, transmittance_before, transmittance_after)`。**全部条目 `contribution` 求和恰等于 `color`**(可加性, demo 与测试均核验)。

## 深度窗 composite_range

`composite_range(deep_samples, z_near, z_far)` 返回半开窗 **`[z_near, z_far)`** 的**独立**合成结果:

- **表面**: 仅当 `z_near <= z < z_far` 计入 —— 恰在 `z_near` 的表面**计入**, 恰在 `z_far` 的表面**排除**。
- **体段**: 先与窗口求交 `(max(z0,z_near), min(z1,z_far))`; 跨边界的体段丢弃窗外部分, 并**按截后长度重新解析积分**(不沿用截前长度)。账本的 `z0/z1` 记录裁剪后的区间, `ids` 保留**原始样本 ID**(重叠小区间列出全部参与 ID)。
- **独立窗口**: 窗内透过率从 1 起算, **窗前遮挡不带入**。这是该窗口"如果它前面什么都没有"的颜色与透过率; 传入窗前样本不会影响窗口结果(窗前样本在裁剪阶段即被排除)。因此窗口结果可被独立检查/复用, 不会出现"带了窗前遮挡却声称是独立窗口"的歧义。
- **拼回**: 沿视线相邻(不重叠)的两个窗口 `(C1,T1)`(近) 与 `(C2,T2)`(远) 用 `combine_results` 拼回:
  - `C = C1 + T1·C2`,`T = T1·T2`。
  - 拼回账本保留两个窗口的原始 ID 与裁剪范围; 后段条目的贡献与透过率统一乘以 `T1`(局部透过率换算为全局透过率), 故接缝处链条连续、全部贡献求和仍**恰等于**拼回颜色。
- **边界情形**:
  - 空窗(窗内无表面、无体段长度)与零宽窗 `z_near == z_far`: 返回零色、`transmittance=1`、空账本。
  - 反向窗 `z_far < z_near`: 抛 `ValidationError`。
  - 体段仅端点落在边界上(如段 `[..., near]` 或 `[far, ...]`)在半开约定下窗内长度为 0, 不产生贡献。
  - 恰在 `z_near` 的表面仍先于"裁剪后从 near 开始"的体段(沿用同 z 处 结束→表面→开始 的顺序)。

## 预乘与合成约定

- 表面 RGBA 为预乘: rgb 已含 alpha, 合成时直接使用, 不再乘 alpha; 不强制 `rgb <= a`(超预乘数据不拒绝, 但按预乘语义解释)。
- 体段端点把深度轴(或窗口内的深度轴)切成不交叠小区间; 每个小区间内重叠体段的 **sigma 相加、emission 相加**, 按均匀介质解析积分:
  - `C = e/σ · (1 − exp(−σ·L))`,`T = exp(−σ·L)`;`σ = 0` 时 `C = e·L, T = 1`。
  - 小 `σ·L` 用 `expm1` 稳定求值, 避免 0/0 精度损失。
- 同一 z 处的顺序: **先结束于 z 的体段区间 → z 处的表面(按 ID 升序)→ 从 z 开始的体段区间**。
- 表面整体按 `(z, id)` 排序, 全部元素 front-to-back 合成: `C += T·c; T *= (1−a)`(表面)或 `T *= T_seg`(体段)。

## 输入上限与错误语义

- **上限**: 每次 `composite()` / `composite_range()` 调用至多 `MAX_ITEMS = 100` 项(按传入条目数计, 不是按窗内条目计), 超出抛 `ItemLimitError`。
- 所有数值输入(含 `z_near`/`z_far`)拒绝 `NaN`/`Infinity`; 整型语义参数(id、窗口边界)拒绝 `bool`; 数值字段同样拒绝 `bool`。
- `sigma < 0` 抛 `NegativeSigmaError`; 负发射、rgba 越界、`z1 < z0`、`z_far < z_near`、错误类型等抛 `ValidationError`。
- 错误消息包含条目上下文(id)与字段名, 可定位; 不静默纠正任何非法输入。
- 两个合成函数均为纯函数: 校验失败在产生结果之前抛出, 不留部分变更; `z0 == z1` 的零长体段是合法 no-op。

## 误差口径

- 体段积分为闭式解析解, 非数值积分; 主要误差来源是 `exp`/`expm1` 与浮点累加(双精度)。
- 对量级正常的输入(σ·L ≲ 数百), 单段相对误差在 **1e-12** 量级; 测试对解析参考使用 `1e-12` 容差, 对极小 σ 的级数极限使用 `1e-9`。
- **分窗拼回**的数学形式与整段积分相同(只是按窗口边界多切了小区间); 任意切点拼回与全量 `composite()` 的偏差在 **1e-11** 内(测试在两个场景 × 6 种切点上核验), demo 的分数切点实测偏差约 1e-17。该误差只来自多一次分段累加, 不用宽容差掩盖边界错误。
- 账本可加性为**精确相等**(颜色由同一组贡献浮点值累加; `combine_results` 的颜色直接由拼回账本求和得到)。
- 为防止"两个错误函数互相抵消", 窗口测试同时对照**三份独立来源**: 测试内逐步手算(仅 `math.exp`)、一份与被测实现结构不同的独立分段解析参考 `reference_range`、以及全量 `composite()`。

## 设计取舍与已知限制

- **窗口是独立结果, 不自动带入窗前遮挡**: 需要整段效果时由调用方显式 `combine_results`(或在单次 `composite_range` 中传入更大窗口)。这是刻意的语义选择, 保证"窗口贡献"无歧义。
- **半开约定只作用于 `composite_range`**; 旧的 `composite(z_far=...)` 保持闭区间远平面语义(`z == z_far` 的表面保留), 接口与行为均未改变。
- **分窗需互不重叠且按视线相邻**: `combine_results` 不校验几何邻接/重叠, 调用方须保证 `first` 在近、`second` 在远; 重叠窗口重复合成不做检测。
- **消光为标量(灰度)**: `sigma` 不随波长变化, 不支持有色消光; 发射为 RGB。
- **重叠体段在小区间内合并**(σ、e 分别相加)后再积分, 账本按合并后的小区间记账(`ids` 列出全部参与条目), 不做逐条目的重叠区归因。
- 仅实现本题契约的合成模型: 不含散射、折射、非均匀介质或体纹理, **不增加新的体采样模型**; 不宣称兼容任何行业深阴影图/deep pixel 标准。
- 表面不强制 `rgb <= a` 的预乘不变式(仅要求各通道 ∈ [0,1])。
- 复杂度 O(N²) 量级(N ≤ 100, 每区间扫描活动体段), 对接口上限足够; 非大数据结构优化目标。

## 文件结构

```
deep_composite/          # 可导入库(核心算法, 无第三方依赖)
  __init__.py            # 公开接口导出
  core.py                # 数据结构、校验、整段/窗口合成与拼回
tests/
  test_composite.py       # 26 个旧单元测试(解析参考独立手算, 全部保留为回归)
  test_composite_range.py # 21 个窗口测试: 手算/独立解析参考/任意分窗重组/边界语义
demo.py                   # 固定输入演示: 整段合成 + 账本可加 + 深度窗 + 分窗拼回 + 拒绝边界
```

# deepcomposite — 双层深度样本的透明合成

离线渲染用的可复用每像素深度样本合成模块。纯 Python 标准库实现，无第三方依赖。

- **Python 版本**：3.14.7（仅用标准库；安装 Python 后无需安装任何其他依赖）
- **运行环境**：Linux 原生，完全离线；核心、测试、演示均不需要网络、账号、数据库或 Docker
- **测试**：`python -m unittest discover -s tests -v`
- **演示**：`python demo.py`（约 1 秒内完成，真实计算并打印一个正常结果与一个实际触发的拒绝）

## 接口

```python
from deepcomposite import (
    Surface, VolumeSegment, composite,
    CompositeResult, LedgerEntry, DeepCompositeError, MAX_ITEMS,
)

result = composite(items, *, z_far=None)
```

- `items`：可迭代对象，最多 `MAX_ITEMS = 100` 项（每像素上限），每项为：
  - `Surface(id, z, rgba)` — 无穷薄表面。`rgba` 为**预乘**颜色，要求 `0 <= r,g,b <= a <= 1`。
  - `VolumeSegment(id, z0, z1, emission, sigma)` — `[z0, z1)` 上的均匀体段，要求 `z1 > z0`；`emission`、`sigma` 为逐通道（RGB）非负系数。
- `z_far`：可选截断远平面。`z > z_far` 的表面被丢弃；体段被裁剪到 `z1 <= z_far`（完全越界的丢弃）。
- 返回 `CompositeResult(color, transmittance, ledger)`：
  - `color`：最终预乘 RGB 三元组；
  - `transmittance`：逐通道透过率三元组（因 `sigma` 逐通道，透过率也是逐通道）；
  - `ledger`：按合成顺序排列的 `LedgerEntry(kind, item_id, z0, z1, contribution, transmittance_before)` 贡献账本。`contribution` 已乘以前方透过率，因此所有账本项之和等于最终色（浮点舍入内）。一条体段与他人的重叠部分会拆成多条账本项，`z0/z1` 标明其覆盖的不交叠小段。

## 合成约定（固定契约）

- 相机朝 `+z` 方向看，`z` 小者近。
- 体段端点把深度轴切成不交叠小段；同一段内重叠体段的 `sigma` 相加、`emission` 相加。
- 小段逐通道积分：`C = e/sigma * (1 - exp(-sigma*L))`，`sigma == 0` 时取 `C = e*L`；小段透过率 `T = exp(-sigma*L)`。
- 表面前到后按 `(z, id)` 排序合成（预乘 over：`C += T*c；T *= 1-a`）。同深度处：先结束于该深度的体段，再合成表面，再开始于该深度的体段。
- 全部颜色均为预乘约定，模块内部不做预乘/非预乘转换。

## 错误语义

- 所有非法输入抛出 `DeepCompositeError`（`ValueError` 子类），消息可定位到具体项与字段，如 `items[3].sigma[1]: negative sigma rejected (got -0.5)`。
- 数值输入拒绝 NaN/Infinity；整数参数（`id`）与浮点参数拒绝 `bool`。
- 拒绝负 `sigma`、负 `emission`、违反预乘约定的 `rgba`（通道越界或 `c > a`）、`z1 <= z0`、超过 100 项、非 Surface/VolumeSegment 项。不做静默纠正或截断。
- 整批先校验后合成：任一非法项使本次调用整体失败，不产生部分结果；`composite` 为纯函数，无副作用。

## 误差口径

- 解析积分用 `math.exp` / `math.expm1`（小 `sigma*L` 时用 `-expm1(-x)` 避免灾难性抵消）。
- 测试以 `math.exp` 独立计算的解析参考为期望值，对量级约 1 的输入采用**绝对容差 1e-12**；典型结果误差在 1e-15 量级。不用宽容差掩盖边界行为：零消光、同深度表面、远平面截断均有精确解析断言。
- 分段不变性（一段拆两段）与账本可加性（账本和 == 最终色）作为补充一致性检查，同样按 1e-12 断言。

## 设计取舍

- 逐通道 `sigma`/`emission`（RGB 三元组）而非标量，以支持有色消光；代价是透过率必须逐通道输出，无法给出单一 alpha。
- 账本按"每个不交叠小段 × 每条体段"记账，使贡献可加核验精确成立；重叠多的场景账本项数随之增加（上界约 `项数 × 断点间隔数`）。
- 同深度多条体段在账本内按 `id` 排序，保证输出确定性。
- 校验集中在 `composite` 入口（而非 dataclass 构造期），以便错误消息携带 `items[i]` 定位信息。

## 已知限制

- 仅实现本题契约的合成数学子集：发射 + 吸收（无散射、无相位函数、无多次弹射），不宣称兼容任何行业体渲染标准。
- 表面为无穷薄、无厚度、无法线/光照模型。
- 单像素接口；不做跨像素加速结构，也不做并行化。
- `z_far` 只截断远端，不提供近平面参数（需要时可在调用前自行过滤）。
- 输入上限 100 项为接口边界；超出即拒绝而非截断。

## 文件

- `deepcomposite/` — 核心模块（`core.py` 全部算法与校验）
- `tests/test_composite.py` — 解析参考、边界与拒绝测试
- `demo.py` — 固定输入演示：正常合成 + 贡献可加核验 + 负 sigma 拒绝

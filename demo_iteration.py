"""deep_composite 迭代演示: 表面样本简化(simplify_surfaces)。

演示内容(全部真实计算, 无预录、无 sleep):
  1. 正常功能: 密集表面样本在误差预算内压缩, 打印压缩前后样本数、
     最坏位置与实际误差, 并把压缩结果送回 composite() 做往返对比;
  2. 边界情况: 超预乘输入折叠颜色越界 -> 候选段被禁用(不钳制颜色)。

运行: python demo_iteration.py   (仅标准库, 毫秒级完成)
"""

from deep_composite import Surface, composite, simplify_surfaces


def build_dense_samples():
    """确定性密集样本: 20 个深度, 每层弱贡献(模拟过密的深度采样)。"""
    return [
        Surface(0.5 * i, (0.018, 0.012, 0.030, 0.04), id=i)
        for i in range(20)
    ]


def main():
    print("=== deep_composite 迭代演示: 表面样本简化 ===")

    samples = build_dense_samples()
    tol = 0.05
    result = simplify_surfaces(samples, tolerance=tol)

    print(f"\n[正常压缩] 20 个表面样本, 误差预算 tolerance={tol}")
    print(f"  压缩后样本数:      {len(result.surfaces)}")
    print(f"  分组终点深度序列:  {result.group_ends}")
    print(f"  实际最大误差:      {result.max_error:.6f} (预算 {tol})")
    print(f"  最坏位置 z:        {result.worst_z}")
    for s in result.surfaces:
        rgba = ", ".join(f"{c:.4f}" for c in s.rgba)
        print(f"    折叠表面 z={s.z:<4} rgba=({rgba}) id={s.id}")

    # 往返: 压缩结果送回旧接口 composite
    before = composite(samples)
    after = composite(result.surfaces)
    print("\n[往返核验] 压缩结果送回 composite():")
    print(f"  压缩前 color = ({before.color[0]:.6f}, {before.color[1]:.6f}, {before.color[2]:.6f}), T = {before.transmittance:.6f}")
    print(f"  压缩后 color = ({after.color[0]:.6f}, {after.color[1]:.6f}, {after.color[2]:.6f}), T = {after.transmittance:.6f}")
    drift = max(abs(before.color[k] - after.color[k]) for k in range(3))
    print(f"  最终色最大通道差 = {drift:.3e} (<= 预算 {tol})")
    assert drift <= tol + 1e-9

    print("\n[边界情况] 超预乘输入, 折叠颜色超出 [0,1] -> 候选段禁用")
    sp = [
        Surface(1.0, (1.0, 0.0, 0.0, 0.0), id=1),  # 合法超预乘: rgb=1 > a=0
        Surface(2.0, (1.0, 0.0, 0.0, 0.0), id=2),
    ]
    res_sp = simplify_surfaces(sp, tolerance=100.0)  # 预算极大也必须拒绝折叠
    print(f"  折叠色 r 分量将达 2.0(越界), 输出样本数 = {len(res_sp.surfaces)}(未压缩)")
    print(f"  原样保留颜色: {[s.rgba[0] for s in res_sp.surfaces]} (未钳制)")
    assert len(res_sp.surfaces) == 2

    print("\n演示完成。")


if __name__ == "__main__":
    main()


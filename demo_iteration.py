"""本轮迭代演示: 表面样本简化 simplify_surfaces。

演示内容(固定输入, 真实计算, 无 sleep):
  1. 一串密集 Surface 在给定误差阈值下被压缩到最少样本, 并送回 composite 复算;
  2. 边界情况: 超预乘表面折叠后颜色越界 -> 压缩候选禁用, 原样保留, 不钳制。

运行: python demo_iteration.py   (纯标准库, 毫秒级完成)
"""

from deep_composite import Surface, composite, simplify_surfaces


def show(title, result):
    print(title)
    print(f"  样本数: {result.input_count} -> {result.output_count}; 分组终点: {result.group_ends}")
    for s in result.surfaces:
        r, g, b, a = s.rgba
        print(f"    z={s.z:<4g} rgba=({r:.4f}, {g:.4f}, {b:.4f}, {a:.4f}) id={s.id!r}")
    print(f"  最坏误差位置 z={result.worst_depth:g}, 实际误差={result.worst_error:.3e}")


def main():
    # 1) 密集的前向散射式层列: 多个相近深度、贡献很小的表面夹着一个较强层。
    surfaces = [
        Surface(1.00, (0.030, 0.010, 0.005, 0.05), id=101),
        Surface(1.05, (0.020, 0.015, 0.010, 0.04), id=102),
        Surface(1.10, (0.025, 0.020, 0.010, 0.06), id=103),
        Surface(1.20, (0.300, 0.120, 0.060, 0.45), id=104),
        Surface(1.25, (0.010, 0.020, 0.040, 0.05), id=105),
        Surface(1.30, (0.015, 0.010, 0.025, 0.04), id=106),
    ]
    tolerance = 0.05
    print(f"阈值 tolerance = {tolerance}")
    result = simplify_surfaces(surfaces, tolerance=tolerance)
    show("压缩结果:", result)

    before = composite(surfaces)
    after = composite(list(result.surfaces))
    print("压缩前后 composite 终态对比:")
    for name, b, a in zip(("r", "g", "b"), before.color, after.color):
        print(f"  color.{name}: {b:.6f} -> {a:.6f}  (|差|={abs(b - a):.3e})")
    print(f"  T       : {before.transmittance:.6f} -> {after.transmittance:.6f}")
    assert result.worst_error <= tolerance
    assert result.surfaces == tuple(sorted(result.surfaces, key=lambda s: (s.z, 0)))

    # 2) 边界情况: 超预乘输入(单通道合法, rgb 远大于 alpha)。
    #    同深度两个这样的表面折叠后 r = 0.9 + 0.9*0.9 = 1.71 > 1,
    #    压缩候选必须禁用, 两个表面原样保留(不钳制颜色)。
    overflow = [
        Surface(2.0, (0.9, 0.9, 0.9, 0.1), id=1),
        Surface(2.0, (0.9, 0.9, 0.9, 0.1), id=2),
    ]
    edge = simplify_surfaces(overflow, tolerance=1.0)
    show("边界情况(超预乘折叠越界, 禁用压缩):", edge)
    assert edge.output_count == 2
    assert edge.surfaces == tuple(overflow)
    print("  折叠后理论 r = 0.9 + 0.9*(1-0.1) = 1.71 > 1: 未钳制, 原样保留。")

    print("OK: 简化结果可送回 composite, 最坏误差在预算内。")


if __name__ == "__main__":
    main()

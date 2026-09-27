"""deep_composite 演示: 固定输入, 真实计算, 离线运行。

演示内容:
  1. 一个确定性场景(彩色消光体重叠 + 同深度表面 + 截断远平面)的合成结果;
  2. RGB 三通道贡献账本的可加性核验(账本逐通道求和 == 最终色);
  3. 一个实际触发的拒绝边界(负 sigma)。

运行: python demo.py   (无第三方依赖, 数毫秒内完成)
"""

from deep_composite import (
    NegativeSigmaError,
    Surface,
    VolumeSegment,
    apply_background,
    composite,
)


def build_scene():
    """确定性示例数据(常量生成, 无随机、无外部输入)。"""
    return [
        # 两段重叠介质: 第一段使用 RGB 消光; 重叠区 emission 和 sigma 均逐通道相加
        VolumeSegment(
            0.0,
            3.0,
            emission=(0.30, 0.10, 0.02),
            sigma=(0.50, 0.20, 0.05),
            id=10,
        ),
        VolumeSegment(2.0, 5.0, emission=(0.05, 0.20, 0.35), sigma=0.25, id=11),
        # 同深度 z=4 的两个表面, 按 ID 升序合成
        Surface(4.0, (0.40, 0.20, 0.10, 0.50), id=21),
        Surface(4.0, (0.05, 0.10, 0.30, 0.25), id=22),
        # 远平面之外的表面, 将被 z_far 丢弃
        Surface(9.0, (0.90, 0.90, 0.90, 1.00), id=23),
    ]


def main():
    print("=== deep_composite 演示: 双层深度样本透明合成 ===")

    scene = build_scene()
    result = composite(scene, z_far=6.0)

    print("\n[正常合成] 5 个样本(2 体段 + 3 表面), z_far=6.0")
    print(f"  最终色(预乘 RGB):   {format_rgb(result.color)}")
    print(f"  三通道透过率 RGB:   {format_rgb(result.transmittance)}")
    bg = (0.02, 0.02, 0.03)
    out = apply_background(result, bg)
    print(f"  叠到背景 {bg} 后:  ({out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f})")

    print("\n[贡献账本] 每个被合成元素一条:")
    print(f"  {'#':>2} {'类型':<8} {'IDs':<10} {'z范围':<14} {'贡献 RGB':<30} {'T前->T后(均为 RGB)'}")
    for i, e in enumerate(result.ledger):
        contrib = format_rgb(e.contribution)
        zr = f"[{e.z0:.1f}, {e.z1:.1f}]"
        t_chain = f"{format_rgb(e.transmittance_before)} -> {format_rgb(e.transmittance_after)}"
        print(
            f"  {i:>2} {e.kind:<8} {str(e.ids):<10} {zr:<14} {contrib:<30} {t_chain}"
        )

    # 可加性核验: 账本贡献之和必须等于最终色
    sums = [0.0, 0.0, 0.0]
    for e in result.ledger:
        for k in range(3):
            sums[k] += e.contribution[k]
    max_diff = max(abs(sums[k] - result.color[k]) for k in range(3))
    print(f"\n[可加性核验] 账本求和 vs 最终色, 最大绝对差 = {max_diff:.3e}")
    assert max_diff < 1e-12, "账本不可加, 实现有误!"
    print("  核验通过: 贡献账本可加。")

    print("\n[拒绝边界] 构造 sigma 为负的体段:")
    try:
        VolumeSegment(
            0.0,
            1.0,
            emission=(0.1, 0.1, 0.1),
            sigma=(0.2, -0.5, 0.1),
            id="bad-smoke",
        )
    except NegativeSigmaError as exc:
        print(f"  已拒绝(NegativeSigmaError): {exc}")

    print("\n演示完成。")


def format_rgb(value):
    return f"({value[0]:.6f}, {value[1]:.6f}, {value[2]:.6f})"


if __name__ == "__main__":
    main()

"""deep_composite 演示: 固定输入, 真实计算, 离线运行。

演示内容:
  1. 一个确定性场景(重叠体段 + 同深度表面 + 截断远平面)的正常合成结果;
  2. 贡献账本的可加性核验(账本求和 == 最终色);
  3. composite_range 深度窗: 窗口独立贡献、窗前遮挡不串入、三窗拼回全量;
  4. 一个实际触发的拒绝边界(负 sigma 与反向窗口)。

运行: python demo.py   (无第三方依赖, 数毫秒内完成)
"""

from deep_composite import (
    NegativeSigmaError,
    Surface,
    ValidationError,
    VolumeSegment,
    apply_background,
    combine_results,
    composite,
    composite_range,
)


def build_scene():
    """确定性示例数据(常量生成, 无随机、无外部输入)。"""
    return [
        # 两段重叠的均匀介质: [0,3] 与 [2,5], 重叠区 sigma/emission 相加
        VolumeSegment(0.0, 3.0, emission=(0.30, 0.10, 0.02), sigma=0.40, id=10),
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
    print(f"  最终色(预乘 RGB): ({result.color[0]:.6f}, {result.color[1]:.6f}, {result.color[2]:.6f})")
    print(f"  整束透过率 T:      {result.transmittance:.6f}")
    bg = (0.02, 0.02, 0.03)
    out = apply_background(result, bg)
    print(f"  叠到背景 {bg} 后:  ({out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f})")

    print("\n[贡献账本] 每个被合成元素一条:")
    print(f"  {'#':>2} {'类型':<8} {'IDs':<10} {'z范围':<14} {'贡献 RGB':<34} {'T前->T后'}")
    for i, e in enumerate(result.ledger):
        contrib = f"({e.contribution[0]:.6f}, {e.contribution[1]:.6f}, {e.contribution[2]:.6f})"
        zr = f"[{e.z0:.1f}, {e.z1:.1f}]"
        print(
            f"  {i:>2} {e.kind:<8} {str(e.ids):<10} {zr:<14} {contrib:<34} "
            f"{e.transmittance_before:.6f} -> {e.transmittance_after:.6f}"
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

    # --- 深度窗: 两段重叠介质 + 恰在窗口边界上的表面 ---
    print("\n[深度窗 composite_range] 场景: A=[0,6] σ=.2 红发射; "
          "B=[2,8] σ=.4 绿发射; 表面 z=2 与 z=6")
    range_scene = [
        VolumeSegment(0.0, 6.0, (0.30, 0.0, 0.0), 0.20, id="A"),
        VolumeSegment(2.0, 8.0, (0.0, 0.50, 0.0), 0.40, id="B"),
        Surface(2.0, (0.10, 0.10, 0.10, 0.50), id="S2"),
        Surface(6.0, (0.20, 0.0, 0.0, 0.50), id="S6"),
    ]
    w1 = composite_range(range_scene, 0.0, 2.0)
    w2 = composite_range(range_scene, 2.0, 6.0)
    w3 = composite_range(range_scene, 6.0, 10.0)
    for name, w in (("W1=[0,2)", w1), ("W2=[2,6)", w2), ("W3=[6,10)", w3)):
        print(
            f"  {name:<10} C=({w.color[0]:.6f}, {w.color[1]:.6f}, "
            f"{w.color[2]:.6f})  T={w.transmittance:.6f}  "
            f"条目={[e.ids[0] if e.kind == 'surface' else e.ids for e in w.ledger]}"
        )
    print("  边界语义: S2 恰在 W1 的 far -> 排除, 恰在 W2 的 near -> 计入; S6 同理。")

    # 窗前放一个不透明遮挡物, 窗口结果必须保持不变(独立性)
    blocked_scene = range_scene + [Surface(1.0, (0.9, 0.9, 0.9, 1.0), id="blocker")]
    w2_blocked = composite_range(blocked_scene, 2.0, 6.0)
    indep_diff = max(abs(w2_blocked.color[i] - w2.color[i]) for i in range(3))
    print(f"  窗前加不透明表面后 W2 结果最大变化 = {indep_diff:.3e} (须为 0)")
    assert indep_diff == 0.0

    # 三窗按 C=C1+T1*C2, T=T1*T2 拼回, 与全量 composite 比较
    joined = combine_results(combine_results(w1, w2), w3)
    full = composite(range_scene)
    cd = max(abs(joined.color[i] - full.color[i]) for i in range(3))
    td = abs(joined.transmittance - full.transmittance)
    print(f"[分窗拼回] vs 全量 composite: 颜色最大差 = {cd:.3e}, 透过率差 = {td:.3e}")
    assert max(cd, td) < 1e-12
    print("  核验通过: 任意分窗重组 == 全量结果(1e-12 内)。")

    print("\n[拒绝边界] 构造 sigma 为负的体段:")
    try:
        VolumeSegment(0.0, 1.0, emission=(0.1, 0.1, 0.1), sigma=-0.5, id="bad-smoke")
    except NegativeSigmaError as exc:
        print(f"  已拒绝(NegativeSigmaError): {exc}")

    print("[拒绝边界] 反向窗口 z_far < z_near:")
    try:
        composite_range(range_scene, 6.0, 2.0)
    except ValidationError as exc:
        print(f"  已拒绝(ValidationError): {exc}")

    print("\n演示完成。")


if __name__ == "__main__":
    main()

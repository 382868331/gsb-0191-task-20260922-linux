"""deep_composite 演示: 固定输入, 真实计算, 离线运行。

演示内容:
  1. 一个确定性场景(重叠体段 + 同深度表面 + 截断远平面)的正常合成结果;
  2. 贡献账本的可加性核验(账本求和 == 最终色);
  3. 一个实际触发的拒绝边界(负 sigma);
  4. 深度窗 composite_range: 半开窗 [near, far) 的独立贡献
     (near 表面计入、far 表面排除、跨边界体段按截长重积分, 窗前遮挡不带入);
  5. 任意分窗后按 C=C1+T1*C2、T=T1*T2 拼回, 与全量结果逐项核对;
  6. 空窗与反向窗的语义/拒绝。

运行: ./.venv/bin/python demo.py   (无第三方依赖, 数毫秒内完成)
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
        # 两段重叠的均匀介质: [0,4] 与 [2,6], 重叠区 sigma/emission 相加
        VolumeSegment(0.0, 4.0, emission=(0.30, 0.10, 0.02), sigma=0.40, id=10),
        VolumeSegment(2.0, 6.0, emission=(0.05, 0.20, 0.35), sigma=0.25, id=11),
        # 同深度 z=4 的两个表面, 按 ID 升序合成
        Surface(4.0, (0.40, 0.20, 0.10, 0.50), id=21),
        Surface(4.0, (0.05, 0.10, 0.30, 0.25), id=22),
        # 远平面之外的表面, 将被 z_far 丢弃
        Surface(9.0, (0.90, 0.90, 0.90, 1.00), id=23),
    ]


def print_result(tag, result):
    print(
        f"  {tag:<26} color=({result.color[0]:.6f}, "
        f"{result.color[1]:.6f}, {result.color[2]:.6f})  T={result.transmittance:.6f}"
    )


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
    assert max_diff == 0.0, "账本不可加, 实现有误!"
    print("  核验通过: 贡献账本可加(精确相等)。")

    print("\n[拒绝边界] 构造 sigma 为负的体段:")
    try:
        VolumeSegment(0.0, 1.0, emission=(0.1, 0.1, 0.1), sigma=-0.5, id="bad-smoke")
    except NegativeSigmaError as exc:
        print(f"  已拒绝(NegativeSigmaError): {exc}")

    # --- 深度窗 -----------------------------------------------------------
    print("\n=== 深度窗 composite_range: 只取深度的一段 ===")
    window_scene = [
        VolumeSegment(0.0, 4.0, (0.30, 0.10, 0.02), 0.40, id=10),
        VolumeSegment(2.0, 6.0, (0.05, 0.20, 0.35), 0.25, id=11),
        Surface(2.0, (0.40, 0.20, 0.10, 0.50), id=21),  # 恰在 near -> 计入
        Surface(3.0, (0.20, 0.00, 0.00, 0.25), id=23),  # 窗内
        Surface(4.0, (0.10, 0.10, 0.10, 0.50), id=22),  # 恰在 far  -> 排除
    ]

    print("\n[独立窗口] 半开窗 [2.0, 4.0): near 表面计入, far 表面排除")
    win = composite_range(window_scene, 2.0, 4.0)
    print_result("窗口 [2,4) 结果", win)
    print("  窗内账本(原始 ID 与裁剪后区间保留, T 为窗内局部透过率):")
    for e in win.ledger:
        contrib = f"({e.contribution[0]:.6f}, {e.contribution[1]:.6f}, {e.contribution[2]:.6f})"
        print(
            f"    {e.kind:<8} ids={str(e.ids):<10} [{e.z0:.1f}, {e.z1:.1f})  "
            f"贡献={contrib:<28} {e.transmittance_before:.6f}->{e.transmittance_after:.6f}"
        )
    assert tuple(e.ids for e in win.ledger if e.kind == "surface") == ((21,), (23,))
    assert all(22 not in e.ids for e in win.ledger), "far 处表面必须排除"
    print("  核验通过: id=21(near) 计入, id=22(far) 排除, 跨界体段按截长重积分。")

    print("\n[窗前遮挡不带入] 窗前再加一个不透明表面, 窗口结果必须不变:")
    occluded = [Surface(0.5, (0.9, 0.9, 0.9, 1.0), id=99)] + window_scene
    win2 = composite_range(occluded, 2.0, 4.0)
    assert win2.color == win.color and win2.transmittance == win.transmittance
    assert all(99 not in e.ids for e in win2.ledger)
    print_result("加窗前遮挡后 [2,4)", win2)
    print("  核验通过: 与无遮挡窗口逐位相等, 遮挡既不进颜色也不进账本。")

    print("\n[任意分窗再拼回] 在 1.7 / 3.0 / 4.8 三处切开(均为任意位置):")
    cuts = [1.7, 3.0, 4.8]
    bounds = [0.0, *cuts, 6.0]
    pieces = [composite_range(window_scene, bounds[i], bounds[i + 1]) for i in range(4)]
    for i, p in enumerate(pieces):
        print_result(f"窗 [{bounds[i]}, {bounds[i+1]})", p)
    joined = pieces[0]
    for p in pieces[1:]:
        joined = combine_results(joined, p)
    full = composite(window_scene)
    print_result("按 C=C1+T1*C2 拼回", joined)
    print_result("全量 composite()  ", full)
    c_diff = max(abs(joined.color[k] - full.color[k]) for k in range(3))
    t_diff = abs(joined.transmittance - full.transmittance)
    print(f"  拼回 vs 全量: 颜色最大差 = {c_diff:.3e}, 透过率差 = {t_diff:.3e}")
    assert c_diff < 1e-12 and t_diff < 1e-12
    # 拼回账本仍保持全局可加
    led_sum = [0.0, 0.0, 0.0]
    for e in joined.ledger:
        for k in range(3):
            led_sum[k] += e.contribution[k]
    assert max(abs(led_sum[k] - joined.color[k]) for k in range(3)) == 0.0
    print("  核验通过: 任意分窗拼回 == 全量结果, 拼回账本精确可加、链条连续。")

    print("\n[空窗 / 零宽窗 / 反向窗]:")
    empty = composite_range(window_scene, 7.0, 9.0)
    print_result("空窗 [7,9)", empty)
    assert empty.color == (0.0, 0.0, 0.0) and empty.transmittance == 1.0 and empty.ledger == ()
    zero_w = composite_range(window_scene, 3.0, 3.0)
    print_result("零宽窗 [3,3)", zero_w)
    assert zero_w.ledger == ()
    try:
        composite_range(window_scene, 4.0, 2.0)
    except ValidationError as exc:
        print(f"  反向窗 [4,2) 已拒绝(ValidationError): {exc}")

    print("\n演示完成。")


if __name__ == "__main__":
    main()

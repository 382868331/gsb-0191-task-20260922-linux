"""deep_composite 演示: 固定输入, 真实计算, 离线运行。

演示内容:
  1. 确定性场景(重叠体段 + 同深度表面 + 截断远平面)的正常合成结果,
     并核验贡献账本可加性;
  2. 零消光体段: C = e*L, T = 1(Beer–Lambert 退化支);
  3. 重叠体段: 端点切分后同段 sigma / emission 相加, 与逐段手算一致;
  4. 遮挡: 不透明表面切断后方体段贡献;
  5. 两个实际触发的拒绝边界(负 sigma 构造失败、composite 收到非法条目)。

运行: python demo.py   (无第三方依赖, 数十毫秒内完成)
"""

import math

from deep_composite import (
    NegativeSigmaError,
    Surface,
    ValidationError,
    VolumeSegment,
    apply_background,
    composite,
)


def print_header(title):
    print(f"\n--- {title} ---")


def main():
    print("=== deep_composite 演示: 深度方向体段 Beer–Lambert 合成 ===")

    # ------------------------------------------------------------------
    # 1) 正常场景: 重叠体段 + 同深度表面 + 远平面截断
    # ------------------------------------------------------------------
    scene = [
        # 两段重叠的均匀介质: [0,3] 与 [2,5], 重叠区 sigma/emission 相加
        VolumeSegment(0.0, 3.0, emission=(0.30, 0.10, 0.02), sigma=0.40, id=10),
        VolumeSegment(2.0, 5.0, emission=(0.05, 0.20, 0.35), sigma=0.25, id=11),
        # 同深度 z=4 的两个表面, 按 ID 升序合成
        Surface(4.0, (0.40, 0.20, 0.10, 0.50), id=21),
        Surface(4.0, (0.05, 0.10, 0.30, 0.25), id=22),
        # 远平面之外的表面, 将被 z_far 丢弃
        Surface(9.0, (0.90, 0.90, 0.90, 1.00), id=23),
    ]
    result = composite(scene, z_far=6.0)

    print_header("正常合成: 5 个样本(2 体段 + 3 表面), z_far=6.0")
    print(f"  最终色(预乘 RGB): ({result.color[0]:.6f}, {result.color[1]:.6f}, {result.color[2]:.6f})")
    print(f"  整束透过率 T:      {result.transmittance:.6f}")
    bg = (0.02, 0.02, 0.03)
    out = apply_background(result, bg)
    print(f"  叠到背景 {bg} 后:  ({out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f})")

    print("  贡献账本(每个被合成元素一条):")
    print(f"  {'#':>2} {'类型':<8} {'IDs':<12} {'z范围':<14} {'T前->T后'}")
    for i, e in enumerate(result.ledger):
        zr = f"[{e.z0:.1f}, {e.z1:.1f}]"
        print(
            f"  {i:>2} {e.kind:<8} {str(e.ids):<12} {zr:<14} "
            f"{e.transmittance_before:.6f} -> {e.transmittance_after:.6f}"
        )

    # 可加性核验: 账本贡献之和必须等于最终色(同一组浮点值, 应精确相等)
    sums = [0.0, 0.0, 0.0]
    for e in result.ledger:
        for k in range(3):
            sums[k] += e.contribution[k]
    max_diff = max(abs(sums[k] - result.color[k]) for k in range(3))
    assert max_diff == 0.0, f"账本不可加, 实现有误! diff={max_diff}"
    print(f"  [核验] 账本求和 == 最终色(最大绝对差 {max_diff:.0e})")

    # 远平面之外的表面 id=23 不得出现在任何账本条目中
    assert not any(23 in e.ids for e in result.ledger)

    # ------------------------------------------------------------------
    # 2) 边界: 零消光体段 C = e*L, T = 1
    # ------------------------------------------------------------------
    print_header("边界: 零消光体段(sigma=0)")
    e0 = (0.25, 0.50, 1.50)
    r0 = composite([VolumeSegment(0.0, 2.0, e0, 0.0, id="zero-sigma")])
    print(f"  输入: e={e0}, L=2, sigma=0")
    print(f"  得到 color=({r0.color[0]:.6f}, {r0.color[1]:.6f}, {r0.color[2]:.6f}), T={r0.transmittance:.6f}")
    # 独立手算期望值(不取自被测实现)
    for got, ec in zip(r0.color, e0):
        assert math.isclose(got, ec * 2.0, abs_tol=1e-12)
    assert r0.transmittance == 1.0
    print("  [核验] color == e*L 且 T == 1")

    # ------------------------------------------------------------------
    # 3) 重叠: 端点切分 + 同段系数相加, 用 Beer–Lambert 闭式逐段手算
    # ------------------------------------------------------------------
    print_header("重叠: 两个体段 [0,2] 与 [1,3]")
    a_seg = VolumeSegment(0.0, 2.0, (0.3, 0.0, 0.0), 0.2, id="A")
    b_seg = VolumeSegment(1.0, 3.0, (0.0, 0.5, 0.0), 0.4, id="B")
    ro = composite([a_seg, b_seg])
    print(f"  账本区间数: {len(ro.ledger)}(端点 0,1,2,3 切成 3 段)")
    for e in ro.ledger:
        print(f"    [{e.z0:.0f}, {e.z1:.0f}] 参与条目={e.ids} T后={e.transmittance_after:.6f}")

    def analytic(e, sigma, length):
        x = sigma * length
        t = math.exp(-x)
        return tuple(ec / sigma * (1.0 - t) for ec in e), t

    # [0,1] 仅 A; [1,2] A+B(sigma=.6, e=(.3,.5,0)); [2,3] 仅 B
    c1, t1 = analytic((0.3, 0.0, 0.0), 0.2, 1.0)
    c2, t2 = analytic((0.3, 0.5, 0.0), 0.6, 1.0)
    c3, t3 = analytic((0.0, 0.5, 0.0), 0.4, 1.0)
    want = [c1[i] + t1 * c2[i] + t1 * t2 * c3[i] for i in range(3)]
    want_t = t1 * t2 * t3
    for i in range(3):
        assert math.isclose(ro.color[i], want[i], abs_tol=1e-12)
    assert math.isclose(ro.transmittance, want_t, abs_tol=1e-12)
    assert set(ro.ledger[1].ids) == {"A", "B"}
    print(f"  [核验] 与逐区间闭式手算一致(容差 1e-12), 重叠区 ID={ro.ledger[1].ids}")

    # ------------------------------------------------------------------
    # 4) 遮挡: 不透明表面之后的体段贡献必须为 0
    # ------------------------------------------------------------------
    print_header("遮挡: 不透明表面切断后方发射体段")
    front_vol = VolumeSegment(0.0, 1.0, (0.1, 0.0, 0.0), 0.0, id="front")
    opaque = Surface(1.0, (0.4, 0.4, 0.4, 1.0), id="wall")
    hidden_vol = VolumeSegment(1.0, 2.0, (2.0, 2.0, 2.0), 0.7, id="hidden")
    rh = composite([hidden_vol, opaque, front_vol])
    print(f"  最终色=({rh.color[0]:.6f}, {rh.color[1]:.6f}, {rh.color[2]:.6f}), T={rh.transmittance:.6f}")
    # 手算: 前体段 (0.1,0,0) + 表面 (0.4,0.4,0.4) + 后方 0
    assert all(math.isclose(g, w, abs_tol=1e-12) for g, w in zip(rh.color, (0.5, 0.4, 0.4)))
    assert rh.transmittance == 0.0
    hidden_entry = [e for e in rh.ledger if "hidden" in e.ids][0]
    assert hidden_entry.contribution == (0.0, 0.0, 0.0)
    print("  [核验] 后方强发射体段贡献恰为 0, T 保持 0")

    # ------------------------------------------------------------------
    # 5) 实际触发的错误边界
    # ------------------------------------------------------------------
    print_header("错误边界(均为真实抛出, 非预录文本)")
    try:
        VolumeSegment(0.0, 1.0, emission=(0.1, 0.1, 0.1), sigma=-0.5, id="bad-smoke")
    except NegativeSigmaError as exc:
        print(f"  构造期拒绝: NegativeSigmaError -> {exc}")

    try:
        composite([Surface(1.0, (0.1, 0.1, 0.1, 0.5)), ("not", "a", "sample")])
    except ValidationError as exc:
        print(f"  合成期拒绝: ValidationError -> {exc}")

    # 失败调用不留状态: 紧接着的合法调用正常工作
    rok = composite([Surface(1.0, (0.5, 0.0, 0.0, 0.5))])
    assert math.isclose(rok.transmittance, 0.5, abs_tol=1e-12)
    print("  [核验] 拒绝后进程内状态干净, 后续合法调用正常")

    print("\n演示完成: 正常结果、零消光/重叠/遮挡边界与实际错误均已展示并核验。")


if __name__ == "__main__":
    main()

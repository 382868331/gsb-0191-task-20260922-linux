"""deep_composite 演示: 固定输入, 真实计算, 离线运行。

演示内容:
  1. 一个确定性场景(重叠体段 + 同深度表面 + 截断远平面)的正常合成结果,
     并用独立手算的逐区间解析参考现场核对;
  2. 边界场景: 零消光重叠体段(T 恒为 1)与不透明表面的遮挡(后方贡献为 0);
  3. 贡献账本的可加性核验(账本求和 == 最终色);
  4. 一个实际触发的拒绝边界(负 sigma)。

运行: python demo.py   (无第三方依赖, 数毫秒内完成)
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


def slab(emission, sigma, length):
    """演示用独立解析参考(与库实现分开写), 用于现场核对。"""
    if sigma == 0.0:
        return tuple(e * length for e in emission), 1.0
    x = sigma * length
    return tuple(e / sigma * (1.0 - math.exp(-x)) for e in emission), math.exp(-x)


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


def analytic_reference():
    """对 build_scene(截断 z_far=6)逐区间独立手算期望结果。"""
    # 端点切分: 0,2,3,4,5,6(表面在 z=4; id=23 的表面在 z_far 之外)
    # [0,2] 仅 v10: e=(.30,.10,.02), sigma=.40
    c1, t1 = slab((0.30, 0.10, 0.02), 0.40, 2.0)
    # [2,3] v10+v11: e 相加=(.35,.30,.37), sigma 相加=.65
    c2, t2 = slab((0.35, 0.30, 0.37), 0.65, 1.0)
    # [3,4] 仅 v11: e=(.05,.20,.35), sigma=.25
    c3, t3 = slab((0.05, 0.20, 0.35), 0.25, 1.0)
    # z=4 两表面按 ID: 21 先 (rgba=.40,.20,.10,.50), 22 后 (.05,.10,.30,.25)
    s21 = (0.40, 0.20, 0.10)
    s22 = (0.05, 0.10, 0.30)
    t_at_4 = t1 * t2 * t3
    color = [c1[k] + t1 * c2[k] + t1 * t2 * c3[k] + t_at_4 * s21[k]
             + t_at_4 * 0.5 * s22[k] for k in range(3)]
    # [4,5] 仅 v11(长度 1); [5,6] 无介质
    c_tail, t_tail = slab((0.05, 0.20, 0.35), 0.25, 1.0)
    t_after_4 = t_at_4 * 0.5 * 0.75
    for k in range(3):
        color[k] += t_after_4 * c_tail[k]
    transmittance = t_after_4 * t_tail
    return tuple(color), transmittance


def demo_normal():
    print("=== deep_composite 演示: 双层深度样本透明合成 ===")

    scene = build_scene()
    result = composite(scene, z_far=6.0)

    print("\n[正常合成] 5 个样本(2 体段 + 3 表面), z_far=6.0")
    print(f"  最终色(预乘 RGB): ({result.color[0]:.6f}, {result.color[1]:.6f}, {result.color[2]:.6f})")
    print(f"  整束透过率 T:      {result.transmittance:.6f}")
    bg = (0.02, 0.02, 0.03)
    out = apply_background(result, bg)
    print(f"  叠到背景 {bg} 后:  ({out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f})")

    want_c, want_t = analytic_reference()
    c_err = max(abs(result.color[k] - want_c[k]) for k in range(3))
    print("\n[独立解析参考核对] 逐区间手算(端点切分 + Beer–Lambert):")
    print(f"  期望 RGB = ({want_c[0]:.6f}, {want_c[1]:.6f}, {want_c[2]:.6f}), "
          f"期望 T = {want_t:.6f}")
    print(f"  RGB 最大偏差 = {c_err:.3e}, T 偏差 = {abs(result.transmittance - want_t):.3e}")
    assert c_err < 1e-12 and abs(result.transmittance - want_t) < 1e-12
    print("  核对通过。")

    print("\n[贡献账本] 每个被合成元素一条:")
    print(f"  {'#':>2} {'类型':<8} {'IDs':<10} {'z范围':<14} {'贡献 RGB':<34} {'T前->T后'}")
    for i, e in enumerate(result.ledger):
        contrib = f"({e.contribution[0]:.6f}, {e.contribution[1]:.6f}, {e.contribution[2]:.6f})"
        zr = f"[{e.z0:.1f}, {e.z1:.1f}]"
        print(
            f"  {i:>2} {e.kind:<8} {str(e.ids):<10} {zr:<14} {contrib:<34} "
            f"{e.transmittance_before:.6f} -> {e.transmittance_after:.6f}"
        )

    sums = [0.0, 0.0, 0.0]
    for e in result.ledger:
        for k in range(3):
            sums[k] += e.contribution[k]
    max_diff = max(abs(sums[k] - result.color[k]) for k in range(3))
    print(f"\n[可加性核验] 账本求和 vs 最终色, 最大绝对差 = {max_diff:.3e}")
    assert max_diff == 0.0, "账本不可加, 实现有误!"
    print("  核验通过: 贡献账本精确可加。")


def demo_boundaries():
    print("\n--- 边界 1: 零消光的重叠体段 ---")
    # [0,2] 红发射 0.3/单位, [1,3] 绿发射 0.5/单位, sigma 全为 0
    res = composite(
        [
            VolumeSegment(0.0, 2.0, (0.3, 0.0, 0.0), 0.0, id="A"),
            VolumeSegment(1.0, 3.0, (0.0, 0.5, 0.0), 0.0, id="B"),
        ]
    )
    print(f"  结果 RGB = ({res.color[0]:.3f}, {res.color[1]:.3f}, {res.color[2]:.3f}), "
          f"T = {res.transmittance:.3f}")
    # 现场手算: 无吸收, T=1; 红 0.3*2=0.6, 绿 0.5*2=1.0
    assert res.color == (0.6, 1.0, 0.0) and res.transmittance == 1.0
    print("  核对: 颜色 = 发射沿深度积分 (0.6, 1.0, 0.0), T 恒为 1。")

    print("\n--- 边界 2: 不透明表面遮挡后方一切 ---")
    # z=2 的不透明表面: 后方 z=3 表面与 [2,4] 亮体段贡献必须为 0
    res = composite(
        [
            VolumeSegment(0.0, 2.0, (0.1, 0.0, 0.0), 0.3, id="front-vol"),
            Surface(2.0, (0.2, 0.2, 0.2, 1.0), id="wall"),
            VolumeSegment(2.0, 4.0, (5.0, 5.0, 5.0), 0.0, id="behind-vol"),
            Surface(3.0, (1.0, 1.0, 1.0, 1.0), id="behind-surf"),
        ]
    )
    behind = [e for e in res.ledger if "behind-vol" in e.ids or "behind-surf" in e.ids]
    print(f"  最终 T = {res.transmittance:.3f}; 后方元素账本条数 = {len(behind)}")
    print(f"  最终 RGB = ({res.color[0]:.6f}, {res.color[1]:.6f}, {res.color[2]:.6f})")
    assert res.transmittance == 0.0
    assert all(e.contribution == (0.0, 0.0, 0.0) for e in behind)
    print("  核对: 不透明表面后 T=0, 后方表面/体段贡献全部为 0。")


def demo_error():
    print("\n[实际错误] 构造 sigma 为负的体段:")
    try:
        VolumeSegment(0.0, 1.0, emission=(0.1, 0.1, 0.1), sigma=-0.5, id="bad-smoke")
    except NegativeSigmaError as exc:
        print(f"  已拒绝(NegativeSigmaError): {exc}")
    else:
        raise AssertionError("负 sigma 必须被拒绝")

    print("[实际错误] composite() 收到非样本条目:")
    try:
        composite([Surface(1.0, (0.1, 0.1, 0.1, 0.5)), ("not", "a", "sample")])
    except ValidationError as exc:
        print(f"  已拒绝(ValidationError): {exc}")
    else:
        raise AssertionError("非法条目必须被拒绝")


def main():
    demo_normal()
    demo_boundaries()
    demo_error()
    print("\n演示完成。")


if __name__ == "__main__":
    main()

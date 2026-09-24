"""deep_composite 单元测试。

解析参考(单段、双重叠、表面合成等)在测试内用 math.exp 独立手算,
不调用被测核心得到期望值; 往返/不变量检查(分段不变、账本可加)仅作补充。
"""

import math
import random
import unittest

from deep_composite import (
    MAX_ITEMS,
    ItemLimitError,
    NegativeSigmaError,
    Surface,
    ValidationError,
    VolumeSegment,
    __all__ as PUBLIC_API,
    apply_background,
    composite,
)

TOL = 1e-12  # 解析参考的浮点容差(见 README 误差口径)


def assert_rgb_close(tc, got, want, tol=TOL):
    tc.assertEqual(len(got), 3)
    for g, w in zip(got, want):
        tc.assertAlmostEqual(g, w, delta=tol)


class TestSingleVolumeSegment(unittest.TestCase):
    """单个体段的解析参考: C = e/sigma*(1-exp(-sigma*L)), T = exp(-sigma*L)。"""

    def test_analytic_reference(self):
        e = (0.3, 0.6, 0.9)
        sigma = 0.5
        z0, z1 = 1.0, 3.0
        length = z1 - z0
        res = composite([VolumeSegment(z0, z1, e, sigma, id=7)])
        x = sigma * length
        want_c = tuple(ec / sigma * (1.0 - math.exp(-x)) for ec in e)
        want_t = math.exp(-x)
        assert_rgb_close(self, res.color, want_c)
        self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)
        self.assertEqual(len(res.ledger), 1)
        entry = res.ledger[0]
        self.assertEqual(entry.kind, "volume")
        self.assertEqual(entry.ids, (7,))
        self.assertEqual((entry.z0, entry.z1), (z0, z1))

    def test_zero_extinction(self):
        """sigma == 0 时 C = e * L, T = 1。"""
        e = (0.25, 0.5, 1.5)  # 发射系数允许 > 1
        z0, z1 = 0.0, 2.0
        res = composite([VolumeSegment(z0, z1, e, 0.0)])
        assert_rgb_close(self, res.color, tuple(ec * (z1 - z0) for ec in e))
        self.assertEqual(res.transmittance, 1.0)

    def test_tiny_sigma_uses_stable_branch(self):
        """极小 sigma 时结果应逼近 e*L(级数极限), 不得因 0/0 崩坏。"""
        e = (1.0, 2.0, 3.0)
        length = 2.0
        res = composite([VolumeSegment(0.0, length, e, 1e-15)])
        for got, ec in zip(res.color, e):
            self.assertAlmostEqual(got, ec * length, delta=1e-9)


class TestOverlappingVolumes(unittest.TestCase):
    """双重叠解析参考: 重叠区 sigma 相加、发射相加, 三段分别积分。"""

    def test_two_overlapping_segments(self):
        # A: [0, 2], e=(0.3, 0.0, 0.0), sigma=0.2
        # B: [1, 3], e=(0.0, 0.5, 0.0), sigma=0.4
        res = composite(
            [
                VolumeSegment(0.0, 2.0, (0.3, 0.0, 0.0), 0.2, id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.5, 0.0), 0.4, id=2),
            ]
        )
        # 独立手算: 区间 [0,1] 仅 A, [1,2] A+B, [2,3] 仅 B
        def seg(e, sigma, length):
            x = sigma * length
            c = tuple(ec / sigma * (1.0 - math.exp(-x)) for ec in e)
            return c, math.exp(-x)

        c1, t1 = seg((0.3, 0.0, 0.0), 0.2, 1.0)
        c2, t2 = seg((0.3, 0.5, 0.0), 0.6, 1.0)
        c3, t3 = seg((0.0, 0.5, 0.0), 0.4, 1.0)
        want = [c1[i] + t1 * c2[i] + t1 * t2 * c3[i] for i in range(3)]
        want_t = t1 * t2 * t3
        assert_rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)
        # 账本应含 3 个体段小区间
        kinds = [e.kind for e in res.ledger]
        self.assertEqual(kinds, ["volume", "volume", "volume"])
        # 中间区间合并了两个原始段
        self.assertEqual(set(res.ledger[1].ids), {1, 2})

    def test_segmentation_invariance(self):
        """分段不变: 一段切成相邻多段, 结果与整段一致(浮点容差内)。"""
        e = (0.4, 0.2, 0.1)
        sigma = 0.7
        whole = composite([VolumeSegment(0.0, 3.0, e, sigma)])
        pieces = composite(
            [
                VolumeSegment(0.0, 1.0, e, sigma, id=1),
                VolumeSegment(1.0, 2.0, e, sigma, id=2),
                VolumeSegment(2.0, 3.0, e, sigma, id=3),
            ]
        )
        assert_rgb_close(self, pieces.color, whole.color, tol=1e-12)
        self.assertAlmostEqual(pieces.transmittance, whole.transmittance, delta=1e-12)


class TestSurfaces(unittest.TestCase):
    def test_single_surface_premultiplied_over(self):
        # 预乘表面: C = c, T = 1 - a
        res = composite([Surface(1.0, (0.2, 0.4, 0.1, 0.5), id=0)])
        assert_rgb_close(self, res.color, (0.2, 0.4, 0.1))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)

    def test_same_depth_surfaces_sorted_by_id(self):
        """同 z 表面按 ID 升序合成(front-to-back)。"""
        # id=1 在前: c1=(0.5,0,0), a1=0.5; id=2 在后: c2=(0,0.5,0), a2=0.5
        s_front = Surface(1.0, (0.5, 0.0, 0.0, 0.5), id=1)
        s_back = Surface(1.0, (0.0, 0.5, 0.0, 0.5), id=2)
        res = composite([s_back, s_front])  # 输入顺序故意打乱
        # 手算: C = c1 + (1-a1)*c2 ; T = (1-a1)(1-a2)
        want = (0.5 + 0.5 * 0.0, 0.0 + 0.5 * 0.5, 0.0)
        assert_rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, 0.25, delta=TOL)
        self.assertEqual([e.ids[0] for e in res.ledger], [1, 2])

    def test_string_ids_sort_after_int_ids(self):
        s_int = Surface(1.0, (0.1, 0.0, 0.0, 0.5), id=5)
        s_str = Surface(1.0, (0.0, 0.1, 0.0, 0.5), id="a")
        res = composite([s_str, s_int])
        self.assertEqual([e.ids[0] for e in res.ledger], [5, "a"])


class TestDepthTieOrdering(unittest.TestCase):
    """同 z: 先结束体段, 再表面, 再开始体段。"""

    def test_volume_end_then_surface_then_volume_start(self):
        # 体段 [0,1] 在 z=1 结束; 表面在 z=1; 体段 [1,2] 在 z=1 开始。
        v1 = VolumeSegment(0.0, 1.0, (1.0, 0.0, 0.0), 0.0, id=1)  # C=(1,0,0), T=1
        surf = Surface(1.0, (0.0, 0.5, 0.0, 0.5), id=2)
        v2 = VolumeSegment(1.0, 2.0, (0.0, 0.0, 1.0), 0.0, id=3)  # C=(0,0,1), T=1
        res = composite([v2, surf, v1])
        # 手算: C = (1,0,0) + 1*(0,0.5,0) + (1*0.5)*(0,0,1) = (1, 0.5, 0.5)
        assert_rgb_close(self, res.color, (1.0, 0.5, 0.5))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)
        self.assertEqual([e.kind for e in res.ledger], ["volume", "surface", "volume"])

    def test_opaque_surface_hides_volume_starting_at_same_z(self):
        v1 = VolumeSegment(0.0, 1.0, (0.2, 0.0, 0.0), 0.0, id=1)
        opaque = Surface(1.0, (0.0, 0.3, 0.0, 1.0), id=2)
        v2 = VolumeSegment(1.0, 5.0, (0.0, 0.0, 9.0), 0.0, id=3)
        res = composite([v2, opaque, v1])
        # 不透明表面后透过率为 0, v2 无贡献
        assert_rgb_close(self, res.color, (0.2, 0.3, 0.0))
        self.assertEqual(res.transmittance, 0.0)


class TestFarPlane(unittest.TestCase):
    def test_truncation_far_plane(self):
        # 体段被截到 z_far; z > z_far 的表面被丢弃
        vol = VolumeSegment(0.0, 10.0, (0.5, 0.5, 0.5), 0.0, id=1)
        near = Surface(1.0, (0.1, 0.0, 0.0, 0.5), id=2)
        far = Surface(8.0, (0.0, 0.0, 0.9, 1.0), id=3)
        res = composite([vol, near, far], z_far=2.0)
        # 手算: 体段 [0,1]: C=0.5; 表面 z=1: c=0.1,a=0.5; 体段 [1,2]: C=0.5
        # C_r = 0.5 + 0.1 + 0.5*0.5 = 0.85 ; C_g = 0.5 + 0 + 0.25 = 0.75 ; T = 0.5
        assert_rgb_close(self, res.color, (0.85, 0.75, 0.75))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)
        self.assertFalse(any(3 in e.ids for e in res.ledger))

    def test_surface_exactly_at_far_plane_kept(self):
        res = composite([Surface(2.0, (0.4, 0.0, 0.0, 0.5), id=1)], z_far=2.0)
        assert_rgb_close(self, res.color, (0.4, 0.0, 0.0))


class TestLedger(unittest.TestCase):
    def test_contributions_sum_to_color(self):
        """账本可加: 全部条目贡献之和 == 最终色。"""
        items = [
            VolumeSegment(0.0, 2.0, (0.3, 0.1, 0.0), 0.4, id=1),
            VolumeSegment(1.0, 3.0, (0.0, 0.2, 0.5), 0.1, id=2),
            Surface(1.5, (0.2, 0.2, 0.2, 0.5), id=3),
            Surface(0.5, (0.1, 0.0, 0.0, 0.25), id=4),
        ]
        res = composite(items)
        sums = [0.0, 0.0, 0.0]
        for entry in res.ledger:
            for i in range(3):
                sums[i] += entry.contribution[i]
        assert_rgb_close(self, sums, res.color, tol=1e-12)
        # 账本链: 每条目的 transmittance_before 为上一条的 after
        for prev, cur in zip(res.ledger, res.ledger[1:]):
            self.assertEqual(prev.transmittance_after, cur.transmittance_before)
        self.assertEqual(res.ledger[0].transmittance_before, 1.0)
        self.assertEqual(res.ledger[-1].transmittance_after, res.transmittance)

    def test_empty_input(self):
        res = composite([])
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_apply_background(self):
        res = composite([Surface(1.0, (0.5, 0.0, 0.0, 0.5), id=0)])
        out = apply_background(res, (0.0, 0.0, 1.0))
        assert_rgb_close(self, out, (0.5, 0.0, 0.5))


class TestValidation(unittest.TestCase):
    def test_negative_sigma_rejected(self):
        with self.assertRaises(NegativeSigmaError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), -0.5)

    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError, msg=f"z={bad}"):
                Surface(bad, (0.1, 0.1, 0.1, 0.5))
            with self.assertRaises(ValidationError, msg=f"sigma={bad}"):
                VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), bad)
            with self.assertRaises(ValidationError, msg=f"z0={bad}"):
                VolumeSegment(bad, 1.0, (0.1, 0.1, 0.1), 0.1)

    def test_bool_rejected_for_numeric_and_id(self):
        with self.assertRaises(ValidationError):
            Surface(True, (0.1, 0.1, 0.1, 0.5))
        with self.assertRaises(ValidationError):
            Surface(1.0, (0.1, 0.1, 0.1, 0.5), id=True)
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), True)

    def test_rgba_range_and_shape(self):
        with self.assertRaises(ValidationError):
            Surface(1.0, (1.5, 0.0, 0.0, 0.5))  # 超 [0,1]
        with self.assertRaises(ValidationError):
            Surface(1.0, (-0.1, 0.0, 0.0, 0.5))
        with self.assertRaises(ValidationError):
            Surface(1.0, (0.1, 0.1, 0.1))  # 长度不为 4

    def test_negative_emission_rejected(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (-0.1, 0.0, 0.0), 0.5)

    def test_inverted_segment_rejected(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(2.0, 1.0, (0.1, 0.1, 0.1), 0.5)

    def test_zero_length_segment_is_noop(self):
        res = composite([VolumeSegment(1.0, 1.0, (1.0, 1.0, 1.0), 1.0)])
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)

    def test_item_limit(self):
        items = [Surface(float(i), (0.1, 0.1, 0.1, 0.1), id=i) for i in range(MAX_ITEMS + 1)]
        with self.assertRaises(ItemLimitError):
            composite(items)
        # 恰好 100 项合法
        res = composite(items[:MAX_ITEMS])
        self.assertEqual(len(res.ledger), MAX_ITEMS)

    def test_wrong_item_type(self):
        with self.assertRaises(ValidationError):
            composite([("not", "a", "sample")])
        with self.assertRaises(ValidationError):
            composite(42)

    def test_z_far_validated(self):
        with self.assertRaises(ValidationError):
            composite([], z_far=float("nan"))

    def test_error_message_locatable(self):
        """错误消息包含字段名与条目上下文, 可定位。"""
        try:
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), -1.0, id="smoke")
        except NegativeSigmaError as exc:
            msg = str(exc)
            self.assertIn("sigma", msg)
            self.assertIn("smoke", msg)
        else:
            self.fail("应抛出 NegativeSigmaError")


class TestZeroExtinctionOverlap(unittest.TestCase):
    """零消光边界: 单段、重叠段以及与有消光段混合。"""

    def test_two_zero_sigma_overlapping_segments(self):
        # A: [0,2] e=(0.3,0,0), B: [1,3] e=(0,0.5,0), 均 sigma=0
        res = composite(
            [
                VolumeSegment(0.0, 2.0, (0.3, 0.0, 0.0), 0.0, id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.5, 0.0), 0.0, id=2),
            ]
        )
        # 解析: 无吸收, T 恒为 1, 颜色 = 发射沿深度的积分
        #   红: 0.3*2 = 0.6 ; 绿: 0.5*2 = 1.0
        assert_rgb_close(self, res.color, (0.6, 1.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual([e.kind for e in res.ledger], ["volume", "volume", "volume"])
        self.assertEqual(set(res.ledger[1].ids), {1, 2})
        for e in res.ledger:
            self.assertEqual(e.transmittance_before, 1.0)
            self.assertEqual(e.transmittance_after, 1.0)

    def test_zero_sigma_mixed_with_absorbing_segment(self):
        # A: [0,2] sigma=0 e=(0.3,0,0); B: [1,3] sigma=0.4 e=(0,0.5,0)
        res = composite(
            [
                VolumeSegment(0.0, 2.0, (0.3, 0.0, 0.0), 0.0, id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.5, 0.0), 0.4, id=2),
            ]
        )
        # 独立手算: [0,1] 仅 A(σ=0); [1,2] A+B(σ=0.4, e 相加); [2,3] 仅 B
        c1, t1 = seg((0.3, 0.0, 0.0), 0.0, 1.0)
        c2, t2 = seg((0.3, 0.5, 0.0), 0.4, 1.0)
        c3, t3 = seg((0.0, 0.5, 0.0), 0.4, 1.0)
        want = [c1[i] + t1 * c2[i] + t1 * t2 * c3[i] for i in range(3)]
        assert_rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, t1 * t2 * t3, delta=TOL)

    def test_all_zero_sigma_scene_transmittance_one(self):
        # 再加一个半透明表面: 体段不衰减, 只有表面影响 T
        res = composite(
            [
                VolumeSegment(0.0, 4.0, (0.1, 0.1, 0.1), 0.0, id=1),
                Surface(2.0, (0.0, 0.0, 0.0, 0.25), id=2),
            ]
        )
        self.assertAlmostEqual(res.transmittance, 0.75, delta=TOL)


def seg(emission, sigma, length):
    """测试用独立解析参考(不调用被测核心)。"""
    if sigma == 0.0:
        return tuple(ec * length for ec in emission), 1.0
    x = sigma * length
    return tuple(ec / sigma * (1.0 - math.exp(-x)) for ec in emission), math.exp(-x)


class TestOcclusion(unittest.TestCase):
    """遮挡: 不透明/半透明表面与体段消光对后方元素的衰减。"""

    def test_opaque_surface_blocks_everything_behind(self):
        v_front = VolumeSegment(0.0, 2.0, (0.1, 0.0, 0.0), 0.3, id=1)
        opaque = Surface(2.0, (0.2, 0.2, 0.2, 1.0), id=2)
        v_back = VolumeSegment(2.0, 4.0, (5.0, 5.0, 5.0), 0.0, id=3)
        s_behind = Surface(3.0, (1.0, 1.0, 1.0, 1.0), id=4)
        res = composite([v_back, s_behind, opaque, v_front])
        c1, t1 = seg((0.1, 0.0, 0.0), 0.3, 2.0)
        want = [c1[i] + t1 * 0.2 for i in range(3)]
        assert_rgb_close(self, res.color, want)
        self.assertEqual(res.transmittance, 0.0)
        # 后方体段与表面仍在账本中(账本为穷举遍历记录), 但贡献恰为 0
        behind = [e for e in res.ledger if any(i in e.ids for i in (3, 4))]
        self.assertTrue(behind)
        for e in behind:
            self.assertEqual(e.contribution, (0.0, 0.0, 0.0))
            self.assertEqual(e.transmittance_before, 0.0)

    def test_translucent_surface_dims_behind_volume(self):
        # a=0.5 的黑表面只衰减不上色; 后方零消光体段贡献被减半
        items = [
            Surface(1.0, (0.0, 0.0, 0.0, 0.5), id=1),
            VolumeSegment(1.0, 3.0, (0.0, 0.0, 0.4), 0.0, id=2),
        ]
        res = composite(items)
        assert_rgb_close(self, res.color, (0.0, 0.0, 0.5 * 0.4 * 2.0))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)

    def test_volume_extinction_occludes_later_surface(self):
        # 体段不发光只吸收: 后方表面颜色乘 exp(-sigma*L)
        vol = VolumeSegment(0.0, 2.0, (0.0, 0.0, 0.0), 0.5, id=1)
        surf = Surface(2.0, (0.6, 0.0, 0.0, 1.0), id=2)
        res = composite([vol, surf])
        t_vol = math.exp(-0.5 * 2.0)
        assert_rgb_close(self, res.color, (t_vol * 0.6, 0.0, 0.0))
        self.assertEqual(res.transmittance, 0.0)

    def test_nested_volume_is_seen_through_outer(self):
        # 外段 [0,4] e_blue=0.2 sigma=0.25; 内段 [1,3] e_red=0.3 sigma=0
        res = composite(
            [
                VolumeSegment(0.0, 4.0, (0.0, 0.0, 0.2), 0.25, id=1),
                VolumeSegment(1.0, 3.0, (0.3, 0.0, 0.0), 0.0, id=2),
            ]
        )
        c1, t1 = seg((0.0, 0.0, 0.2), 0.25, 1.0)
        c2, t2 = seg((0.3, 0.0, 0.2), 0.25, 2.0)
        c3, t3 = seg((0.0, 0.0, 0.2), 0.25, 1.0)
        want = [c1[i] + t1 * c2[i] + t1 * t2 * c3[i] for i in range(3)]
        assert_rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, t1 * t2 * t3, delta=TOL)
        # 内段被外段消光遮挡: 它的红贡献恰为 T([0,1]) * (0.3 在内区间的积分)
        red_inner_only = t1 * c2[0]
        ledger_red_inner = sum(
            e.contribution[0] for e in res.ledger if 2 in e.ids
        )
        self.assertAlmostEqual(ledger_red_inner, red_inner_only, delta=TOL)

    def test_fully_absorbing_volume_black_behind(self):
        # 极厚强消光体段: 后方表面贡献数值上为 0
        vol = VolumeSegment(0.0, 50.0, (0.0, 0.0, 0.0), 100.0, id=1)
        surf = Surface(50.0, (1.0, 1.0, 1.0, 1.0), id=2)
        res = composite([vol, surf])
        assert_rgb_close(self, res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 0.0)


class TestOverlapGeometry(unittest.TestCase):
    """重叠几何: 包含、三方重叠、端点相接。"""

    def test_nested_segments_intervals(self):
        res = composite(
            [
                VolumeSegment(0.0, 4.0, (0.1, 0.0, 0.0), 0.2, id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.2, 0.0), 0.3, id=2),
            ]
        )
        self.assertEqual(
            [(e.z0, e.z1) for e in res.ledger], [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)]
        )
        self.assertEqual([e.ids for e in res.ledger], [(1,), (1, 2), (1,)])
        c1, t1 = seg((0.1, 0.0, 0.0), 0.2, 1.0)
        c2, t2 = seg((0.1, 0.2, 0.0), 0.5, 2.0)
        c3, t3 = seg((0.1, 0.0, 0.0), 0.2, 1.0)
        want = [c1[i] + t1 * c2[i] + t1 * t2 * c3[i] for i in range(3)]
        assert_rgb_close(self, res.color, want)

    def test_three_way_overlap(self):
        # [0,3] / [1,4] / [2,5], 全部零消光, 便于逐区间核对
        res = composite(
            [
                VolumeSegment(0.0, 3.0, (0.1, 0.0, 0.0), 0.0, id=1),
                VolumeSegment(1.0, 4.0, (0.0, 0.1, 0.0), 0.0, id=2),
                VolumeSegment(2.0, 5.0, (0.0, 0.0, 0.1), 0.0, id=3),
            ]
        )
        expected_intervals = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        self.assertEqual(
            [(e.z0, e.z1) for e in res.ledger], expected_intervals
        )
        self.assertEqual(
            [set(e.ids) for e in res.ledger],
            [{1}, {1, 2}, {1, 2, 3}, {2, 3}, {3}],
        )
        # 红: 0.1*3=0.3 ; 绿: 0.1*3=0.3 ; 蓝: 0.1*3=0.3
        assert_rgb_close(self, res.color, (0.3, 0.3, 0.3))
        self.assertEqual(res.transmittance, 1.0)

    def test_touching_segments_do_not_merge(self):
        res = composite(
            [
                VolumeSegment(0.0, 1.0, (0.2, 0.0, 0.0), 0.5, id=1),
                VolumeSegment(1.0, 2.0, (0.2, 0.0, 0.0), 0.5, id=2),
            ]
        )
        self.assertEqual(len(res.ledger), 2)
        self.assertEqual([e.ids for e in res.ledger], [(1,), (2,)])
        # 相接与整段完全等价(分段不变性的另一形式)
        whole = composite([VolumeSegment(0.0, 2.0, (0.2, 0.0, 0.0), 0.5, id=9)])
        assert_rgb_close(self, res.color, whole.color)
        self.assertAlmostEqual(
            res.transmittance, whole.transmittance, delta=TOL
        )


def _slab_reference(items, z_far=None):
    """独立枚举参考: 按事件坐标逐开区间 + 坐标点表面推进。

    与被测实现分开手写: 区间归属用中点包含判定, 小区间解析积分,
    表面在区间之后按 (int 优先, id) 顺序应用。
    """
    surfaces = []
    spans = []
    for it in items:
        if isinstance(it, Surface):
            if z_far is None or it.z <= z_far:
                surfaces.append(it)
        else:
            if it.z0 == it.z1:
                continue
            z1 = it.z1 if z_far is None else min(it.z1, z_far)
            if z1 <= it.z0:
                continue
            spans.append((it.z0, z1, it.emission, it.sigma, it.id))

    coords = sorted({z for a, b, *_ in spans for z in (a, b)} | {s.z for s in surfaces})
    surf_at = {}
    for s in surfaces:
        surf_at.setdefault(s.z, []).append(s)

    color = [0.0, 0.0, 0.0]
    t = 1.0
    for k, z in enumerate(coords):
        if k > 0:
            a, b = coords[k - 1], z
            if b > a:
                mid = (a + b) / 2.0
                sig = 0.0
                emi = [0.0, 0.0, 0.0]
                for za, zb, e, s, _i in spans:
                    if za <= mid <= zb:  # 端点相接只可能落在单点, 区间长度为 0
                        sig += s
                        emi[0] += e[0]
                        emi[1] += e[1]
                        emi[2] += e[2]
                c_seg, t_seg = seg((emi[0], emi[1], emi[2]), sig, b - a)
                for ch in range(3):
                    color[ch] += t * c_seg[ch]
                t *= t_seg
        for s in sorted(surf_at.get(z, ()), key=lambda x: (0, x.id) if isinstance(x.id, int) else (1, x.id)):
            for ch in range(3):
                color[ch] += t * s.rgba[ch]
            t *= 1.0 - s.rgba[3]
    return (color[0], color[1], color[2]), t


class TestEnumeratedReference(unittest.TestCase):
    """小规模随机枚举: 与独立 slab 参考逐场景核对(含零消光、遮挡、重叠)。"""

    def test_random_scenes_against_slab_reference(self):
        rng = random.Random(20260924)
        sigmas = [0.0, 0.0, 0.2, 0.75, 2.0]
        emissions = [0.0, 0.15, 0.4, 1.2]
        depths = [0.0, 1.0, 2.0, 3.0, 4.0]
        alphas = [0.0, 0.25, 0.5, 1.0]
        for scene_idx in range(60):
            items = []
            for vid in range(rng.randrange(1, 5)):
                a = rng.choice(depths)
                b = a + rng.choice([0.0, 1.0, 2.0])
                e = tuple(rng.choice(emissions) for _ in range(3))
                items.append(VolumeSegment(a, b, e, rng.choice(sigmas), id=f"v{vid}"))
            for sid in range(rng.randrange(0, 4)):
                rgb = tuple(rng.choice([0.0, 0.2, 0.6]) for _ in range(3))
                items.append(
                    Surface(rng.choice(depths), rgb + (rng.choice(alphas),), id=sid)
                )
            z_far = rng.choice([None, None, 2.0, 5.0])
            res = composite(items, z_far=z_far)
            want_c, want_t = _slab_reference(items, z_far)
            assert_rgb_close(self, res.color, want_c, tol=1e-11)
            self.assertAlmostEqual(res.transmittance, want_t, delta=1e-11)
            # 每个枚举场景都核验账本可加与透过率链
            sums = [0.0, 0.0, 0.0]
            for e in res.ledger:
                for ch in range(3):
                    sums[ch] += e.contribution[ch]
            assert_rgb_close(self, sums, res.color, tol=1e-12)
            for prev, cur in zip(res.ledger, res.ledger[1:]):
                self.assertEqual(prev.transmittance_after, cur.transmittance_before)


class TestAdditionalValidation(unittest.TestCase):
    def test_z_far_inf_rejected(self):
        with self.assertRaises(ValidationError):
            composite([Surface(1.0, (0.1, 0.1, 0.1, 0.5))], z_far=float("inf"))

    def test_id_wrong_types(self):
        for bad in (1.5, None, (1,), [1]):
            with self.assertRaises(ValidationError, msg=f"id={bad!r}"):
                Surface(1.0, (0.1, 0.1, 0.1, 0.5), id=bad)
            with self.assertRaises(ValidationError, msg=f"id={bad!r}"):
                VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), 0.1, id=bad)

    def test_emission_shape_and_component_types(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1), 0.1)  # 长度不为 3
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, "rgb", 0.1)
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, "x", 0.1), 0.1)

    def test_rgba_component_type(self):
        with self.assertRaises(ValidationError):
            Surface(1.0, (0.1, 0.1, None, 0.5))

    def test_apply_background_validation(self):
        res = composite([Surface(1.0, (0.5, 0.0, 0.0, 0.5))])
        with self.assertRaises(ValidationError):
            apply_background(res, (0.1, 0.1))  # 长度不为 3
        with self.assertRaises(ValidationError):
            apply_background(res, (0.1, float("nan"), 0.1))

    def test_accepts_generator_and_is_pure(self):
        gen = (
            VolumeSegment(float(i), float(i + 1), (0.1, 0.0, 0.0), 0.0, id=i)
            for i in range(3)
        )
        res = composite(gen)
        assert_rgb_close(self, res.color, (0.3, 0.0, 0.0))
        items = [
            VolumeSegment(0.0, 2.0, (0.3, 0.1, 0.0), 0.4, id=1),
            Surface(1.0, (0.2, 0.2, 0.2, 0.5), id=2),
        ]
        snapshot = list(items)
        first = composite(items)
        second = composite(items)  # 重复调用结果一致, 输入不被破坏
        self.assertEqual(first.color, second.color)
        self.assertEqual(first.transmittance, second.transmittance)
        self.assertEqual(items, snapshot)
        # 输入条目本身不可变
        with self.assertRaises(AttributeError):
            items[0].sigma = 9.0

    def test_public_api_surface(self):
        # 公开入口稳定: README 列出的名字全部可导入
        for name in (
            "Surface", "VolumeSegment", "composite", "apply_background",
            "CompositeResult", "LedgerEntry", "MAX_ITEMS",
            "DeepCompositeError", "ValidationError",
            "NegativeSigmaError", "ItemLimitError",
        ):
            self.assertIn(name, PUBLIC_API)


if __name__ == "__main__":
    unittest.main()

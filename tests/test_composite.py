"""deep_composite 单元测试。

解析参考(单段、双重叠、表面合成等)在测试内用 math.exp 独立手算,
不调用被测核心得到期望值; 往返/不变量检查(分段不变、账本可加)仅作补充。
"""

import itertools
import math
import unittest

from deep_composite import (
    MAX_ITEMS,
    ItemLimitError,
    NegativeSigmaError,
    Surface,
    ValidationError,
    VolumeSegment,
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


class TestBeerLambert(unittest.TestCase):
    """Beer–Lambert 行为: 纯吸收、发射被前方介质/表面衰减、多段透过率连乘。"""

    def test_pure_absorber_zero_emission(self):
        """e=0, sigma>0: 颜色贡献为 0, T = exp(-sigma*L)。"""
        sigma = math.log(2.0)  # 使 T 恰为 1/2
        res = composite([VolumeSegment(0.0, 1.0, (0.0, 0.0, 0.0), sigma, id=1)])
        assert_rgb_close(self, res.color, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)

    def test_emission_behind_absorber_is_attenuated(self):
        """后方发射体段的贡献被前方吸收段按 T 衰减。"""
        ln2 = math.log(2.0)
        # A: [0,1] 纯吸收 T=1/2; B: [1,2] 零消光发射 C=(0.6,0,0)
        res = composite(
            [
                VolumeSegment(0.0, 1.0, (0.0, 0.0, 0.0), ln2, id=1),
                VolumeSegment(1.0, 2.0, (0.6, 0.0, 0.0), 0.0, id=2),
            ]
        )
        # 手算: color = T_A * C_B = 0.5 * 0.6 = 0.3; T = 0.5
        assert_rgb_close(self, res.color, (0.3, 0.0, 0.0))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)

    def test_chained_absorbers_transmittance_multiplies(self):
        """两段不相交吸收段: T = exp(-(sigma1*L1 + sigma2*L2))。"""
        ln2, ln3 = math.log(2.0), math.log(3.0)
        res = composite(
            [
                VolumeSegment(0.0, 1.0, (0.0, 0.0, 0.0), ln2, id=1),
                VolumeSegment(2.0, 3.0, (0.0, 0.0, 0.0), ln3, id=2),
            ]
        )
        # 中间 [1,2] 为真空, 不改变 T; 总 T = 1/2 * 1/3 = 1/6
        self.assertAlmostEqual(res.transmittance, 1.0 / 6.0, delta=TOL)
        assert_rgb_close(self, res.color, (0.0, 0.0, 0.0))

    def test_surface_then_emitting_absorbing_volume(self):
        """半透明表面后接有消光发射段: 手算解析值。"""
        # 表面: c=(0.2,0.1,0), a=0.5 -> T=0.5
        # 体段 [1,3]: sigma=0.5, e=(0,0,0.8), L=2, x=1
        res = composite(
            [
                Surface(1.0, (0.2, 0.1, 0.0, 0.5), id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.0, 0.8), 0.5, id=2),
            ]
        )
        c_vol = 0.8 / 0.5 * (1.0 - math.exp(-1.0))  # 1.6*(1-e^-1)
        want_b = 0.5 * c_vol
        self.assertAlmostEqual(res.color[0], 0.2, delta=TOL)
        self.assertAlmostEqual(res.color[1], 0.1, delta=TOL)
        self.assertAlmostEqual(res.color[2], want_b, delta=TOL)
        self.assertAlmostEqual(res.transmittance, 0.5 * math.exp(-1.0), delta=TOL)


class TestOcclusion(unittest.TestCase):
    """遮挡: 不透明/半透明表面切断后方贡献。"""

    def test_opaque_surface_hides_absorbing_emitting_volume(self):
        """不透明表面后方的 sigma>0 发射段贡献严格为 0, T 保持 0。"""
        # 前方零消光发射段 [0,1]: C=(0.1,0,0)
        # z=1 不透明表面: c=(0.4,0.4,0.4), a=1
        # 后方强发射吸收段 [1,2]: 必须无贡献
        res = composite(
            [
                VolumeSegment(0.0, 1.0, (0.1, 0.0, 0.0), 0.0, id=1),
                Surface(1.0, (0.4, 0.4, 0.4, 1.0), id=2),
                VolumeSegment(1.0, 2.0, (2.0, 2.0, 2.0), 0.7, id=3),
            ]
        )
        # 手算: (0.1,0,0) + (0.4,0.4,0.4) + 0 = (0.5,0.4,0.4)
        assert_rgb_close(self, res.color, (0.5, 0.4, 0.4))
        self.assertEqual(res.transmittance, 0.0)
        back = [e for e in res.ledger if 3 in e.ids]
        self.assertEqual(len(back), 1)
        assert_rgb_close(self, back[0].contribution, (0.0, 0.0, 0.0))
        self.assertEqual(back[0].transmittance_before, 0.0)
        self.assertEqual(back[0].transmittance_after, 0.0)

    def test_semi_transparent_surface_partially_attenuates_background_volume(self):
        """a=0.5 表面后方的零消光发射段贡献乘 (1-a)=0.5。"""
        # 表面 z=1 c=(0,0,0.2) a=0.5; 体段 [1,2] e=(0.8,0,0) sigma=0
        res = composite(
            [
                Surface(1.0, (0.0, 0.0, 0.2, 0.5), id=1),
                VolumeSegment(1.0, 2.0, (0.8, 0.0, 0.0), 0.0, id=2),
            ]
        )
        # 手算: color = (0,0,0.2) + 0.5*(0.8,0,0) = (0.4,0,0.2); T=0.5
        assert_rgb_close(self, res.color, (0.4, 0.0, 0.2))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)

    def test_opaque_surface_also_hides_later_surface(self):
        s_front = Surface(1.0, (0.3, 0.3, 0.3, 1.0), id=1)
        s_back = Surface(2.0, (0.9, 0.9, 0.9, 1.0), id=2)
        res = composite([s_back, s_front])
        assert_rgb_close(self, res.color, (0.3, 0.3, 0.3))
        self.assertEqual(res.transmittance, 0.0)
        back = [e for e in res.ledger if e.ids == (2,)][0]
        assert_rgb_close(self, back.contribution, (0.0, 0.0, 0.0))


class TestOverlapCoefficientAddition(unittest.TestCase):
    """端点切分后, 同段介质 sigma / emission 分别相加(三重重叠、零消光混合)。"""

    def test_three_way_overlap(self):
        # A [0,3] e=(.1,0,0)  sigma=.1
        # B [1,3] e=(0,.2,0)  sigma=.2
        # C [2,4] e=(0,0,.3)  sigma=.3
        res = composite(
            [
                VolumeSegment(0.0, 3.0, (0.1, 0.0, 0.0), 0.1, id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.2, 0.0), 0.2, id=2),
                VolumeSegment(2.0, 4.0, (0.0, 0.0, 0.3), 0.3, id=3),
            ]
        )

        def seg(e, sigma, length):
            x = sigma * length
            c = tuple(ec / sigma * (1.0 - math.exp(-x)) for ec in e)
            return c, math.exp(-x)

        # 手算四个小区间:
        c1, t1 = seg((0.1, 0.0, 0.0), 0.1, 1.0)       # [0,1] 仅 A
        c2, t2 = seg((0.1, 0.2, 0.0), 0.3, 1.0)       # [1,2] A+B
        c3, t3 = seg((0.1, 0.2, 0.3), 0.6, 1.0)       # [2,3] A+B+C
        c4, t4 = seg((0.0, 0.0, 0.3), 0.3, 1.0)       # [3,4] 仅 C
        f21, f321 = t1, t1 * t2
        f4321 = t1 * t2 * t3
        want = tuple(
            c1[i] + f21 * c2[i] + f321 * c3[i] + f4321 * c4[i] for i in range(3)
        )
        assert_rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, t1 * t2 * t3 * t4, delta=TOL)
        self.assertEqual(len(res.ledger), 4)
        self.assertEqual(set(res.ledger[2].ids), {1, 2, 3})

    def test_coincident_segments_merge_completely(self):
        """完全重合的两段: 合并为一个账本小区间, sigma 相加。"""
        res = composite(
            [
                VolumeSegment(0.0, 2.0, (0.3, 0.0, 0.0), 0.3, id=1),
                VolumeSegment(0.0, 2.0, (0.0, 0.0, 0.5), 0.4, id=2),
            ]
        )
        vol_entries = [e for e in res.ledger if e.kind == "volume"]
        self.assertEqual(len(vol_entries), 1)
        entry = vol_entries[0]
        self.assertEqual((entry.z0, entry.z1), (0.0, 2.0))
        self.assertEqual(set(entry.ids), {1, 2})
        # 手算: sigma=.7, e=(.3,0,.5), L=2, x=1.4
        x = 1.4
        want_c = tuple(ec / 0.7 * (1.0 - math.exp(-x)) for ec in (0.3, 0.0, 0.5))
        assert_rgb_close(self, res.color, want_c)
        self.assertAlmostEqual(res.transmittance, math.exp(-x), delta=TOL)

    def test_zero_extinction_overlapping_absorbing(self):
        """零消光段与吸收段重叠: 合并小区间 sigma>0, 发射一起受 Beer 衰减。"""
        ln2 = math.log(2.0)
        # A [0,2] e=(.5,0,0) sigma=0 ; B [1,3] e=(0,.7,0) sigma=ln2
        res = composite(
            [
                VolumeSegment(0.0, 2.0, (0.5, 0.0, 0.0), 0.0, id=1),
                VolumeSegment(1.0, 3.0, (0.0, 0.7, 0.0), ln2, id=2),
            ]
        )

        def seg(e, sigma, length):
            x = sigma * length
            c = tuple(ec / sigma * (1.0 - math.exp(-x)) for ec in e)
            return c, math.exp(-x)

        c1, t1 = ((0.5, 0.0, 0.0), 1.0)                 # [0,1] 仅 A: C=e*L
        c2, t2 = seg((0.5, 0.7, 0.0), ln2, 1.0)         # [1,2] 合并
        c3, t3 = seg((0.0, 0.7, 0.0), ln2, 1.0)         # [2,3] 仅 B
        want = tuple(c1[i] + t1 * c2[i] + t1 * t2 * c3[i] for i in range(3))
        assert_rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, t1 * t2 * t3, delta=TOL)
        # 中间区间同时列出两个参与条目
        self.assertEqual(set(res.ledger[1].ids), {1, 2})

    def test_endpoint_touch_is_not_overlap(self):
        """仅端点相接不算重叠: 各自独立记账, T 连乘。"""
        ln2 = math.log(2.0)
        res = composite(
            [
                VolumeSegment(0.0, 1.0, (0.0, 0.0, 0.0), ln2, id=1),
                VolumeSegment(1.0, 2.0, (0.0, 0.0, 0.0), ln2, id=2),
            ]
        )
        entries = [e for e in res.ledger if e.kind == "volume"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].ids, (1,))
        self.assertEqual(entries[1].ids, (2,))
        self.assertAlmostEqual(res.transmittance, 0.25, delta=TOL)


# ---------------------------------------------------------------------------
# 独立参考实现(事件扫描, 与核心的"每区间全量扫描"结构不同), 仅用于枚举核对
# ---------------------------------------------------------------------------

def _reference_composite(items, z_far=None):
    """测试专用独立参考: 坐标事件 + 活动集合增量维护, 不调用被测核心。"""
    if z_far is not None:
        if not isinstance(z_far, (int, float)) or isinstance(z_far, bool):
            raise ValueError("bad z_far")
    surfaces = []
    starts = {}
    ends = {}
    segs = {}
    coords = set()
    for it in items:
        if isinstance(it, Surface):
            if z_far is None or it.z <= z_far:
                surfaces.append(it)
                coords.add(it.z)
        elif isinstance(it, VolumeSegment):
            z0, z1 = it.z0, it.z1
            if z0 == z1:
                continue
            if z_far is not None:
                z1 = min(z1, z_far)
                if z1 <= z0:
                    continue
            starts.setdefault(z0, []).append(it.id)
            ends.setdefault(z1, []).append(it.id)
            segs[it.id] = (it.emission, it.sigma)
            coords.add(z0)
            coords.add(z1)
        else:
            raise ValueError("bad item")

    surfaces_at = {}
    for s in surfaces:
        surfaces_at.setdefault(s.z, []).append(s)
    for z in surfaces_at:
        surfaces_at[z].sort(
            key=lambda s: (0, s.id) if isinstance(s.id, int) else (1, s.id)
        )

    ordered = sorted(coords)
    color = [0.0, 0.0, 0.0]
    t = 1.0
    active = {}
    for k, z in enumerate(ordered):
        if k > 0:
            for iid in starts.get(ordered[k - 1], ()):
                active[iid] = segs[iid]
            a, b = ordered[k - 1], z
            if b > a and active:
                sigma = sum(v[1] for v in active.values())
                emi = [0.0, 0.0, 0.0]
                for e, _s in active.values():
                    for c in range(3):
                        emi[c] += e[c]
                length = b - a
                x = sigma * length
                tseg = math.exp(-x)
                if x == 0.0:
                    cseg = [emi[c] * length for c in range(3)]
                else:
                    cseg = [emi[c] / sigma * (1.0 - tseg) for c in range(3)]
                for c in range(3):
                    color[c] += t * cseg[c]
                t *= tseg
            for iid in ends.get(z, ()):
                active.pop(iid, None)
        for s in surfaces_at.get(z, ()):
            r, g, bl, a = s.rgba
            color[0] += t * r
            color[1] += t * g
            color[2] += t * bl
            t *= 1.0 - a
    return (color[0], color[1], color[2]), t


class TestEnumeratedScenes(unittest.TestCase):
    """小规模枚举: 输入排列不变, 且与独立事件扫描参考一致。"""

    def test_permutations_against_independent_reference(self):
        pool = [
            Surface(0.5, (0.2, 0.0, 0.0, 0.5), id="s0"),
            Surface(1.0, (0.0, 0.3, 0.0, 0.25), id="s1"),
            Surface(2.0, (0.1, 0.1, 0.1, 1.0), id="s2"),
            VolumeSegment(0.0, 1.5, (0.3, 0.0, 0.0), 0.4, id="v0"),
            VolumeSegment(1.0, 2.5, (0.0, 0.2, 0.1), 0.0, id="v1"),
            VolumeSegment(2.0, 3.0, (0.05, 0.05, 0.05), 0.7, id="v2"),
        ]
        z_fars = (None, 1.5, 0.75)
        scenes = 0
        for size in range(1, 5):
            for subset in itertools.combinations(pool, size):
                for perm in itertools.permutations(subset):
                    items = list(perm)
                    for z_far in z_fars:
                        got = composite(items, z_far=z_far)
                        want_c, want_t = _reference_composite(items, z_far=z_far)
                        assert_rgb_close(self, got.color, want_c)
                        self.assertAlmostEqual(
                            got.transmittance, want_t, delta=1e-12
                        )
                        # 账本可加性(精确)与透过率单调不增
                        sums = [0.0, 0.0, 0.0]
                        prev_after = 1.0
                        for e in got.ledger:
                            self.assertEqual(e.transmittance_before, prev_after)
                            self.assertLessEqual(e.transmittance_after, prev_after + 1e-15)
                            for c in range(3):
                                sums[c] += e.contribution[c]
                            prev_after = e.transmittance_after
                        assert_rgb_close(self, tuple(sums), got.color, tol=1e-12)
                        self.assertEqual(prev_after, got.transmittance)
                        scenes += 1
        self.assertGreater(scenes, 1000)  # 确认枚举确实有规模, 非空转


class TestBoundary(unittest.TestCase):
    def test_thick_medium_transmittance_underflows(self):
        """sigma*L=1000: T=0, 自发射收敛到 e/sigma, 不抛异常。"""
        res = composite([VolumeSegment(0.0, 1.0, (3.0, 2.0, 1.0), 1000.0)])
        self.assertEqual(res.transmittance, 0.0)
        assert_rgb_close(self, res.color, (0.003, 0.002, 0.001), tol=1e-12)

    def test_z_far_before_everything_drops_all(self):
        res = composite(
            [
                Surface(1.0, (0.5, 0.5, 0.5, 1.0)),
                VolumeSegment(0.0, 2.0, (1.0, 1.0, 1.0), 0.5),
            ],
            z_far=-1.0,
        )
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_generator_input_consumed_once(self):
        gen = (Surface(float(i), (0.1, 0.1, 0.1, 0.5), id=i) for i in range(3))
        res = composite(gen)
        self.assertEqual(len(res.ledger), 3)

    def test_no_shared_state_after_failed_call(self):
        items = [Surface(float(i), (0.1, 0.1, 0.1, 0.1), id=i) for i in range(3)]
        with self.assertRaises(ItemLimitError):
            composite(items * (MAX_ITEMS // 3 + 1))
        res = composite(items)  # 失败调用后功能不受影响
        self.assertEqual(len(res.ledger), 3)


class TestExtraValidation(unittest.TestCase):
    def test_bool_id_rejected_for_volume(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), 0.5, id=False)

    def test_emission_shape_and_type(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1), 0.5)  # 长度不为 3
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, ("x", 0.1, 0.1), 0.5)

    def test_sigma_string_rejected(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), "0.5")

    def test_rgba_component_string_rejected(self):
        with self.assertRaises(ValidationError):
            Surface(1.0, (0.1, "x", 0.1, 0.5))

    def test_bool_z_far_rejected(self):
        with self.assertRaises(ValidationError):
            composite([], z_far=True)

    def test_apply_background_validated(self):
        res = composite([Surface(1.0, (0.5, 0.0, 0.0, 0.5))])
        with self.assertRaises(ValidationError):
            apply_background(res, (0.1, 0.1))        # 长度不为 3
        with self.assertRaises(ValidationError):
            apply_background(res, (0.1, -0.1, 0.1))  # 负值


if __name__ == "__main__":
    unittest.main()

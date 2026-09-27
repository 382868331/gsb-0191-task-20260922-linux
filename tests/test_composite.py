"""deep_composite 单元测试。

解析参考(单段、双重叠、表面合成等)在测试内用 math.exp 独立手算,
不调用被测核心得到期望值; 往返/不变量检查(分段不变、账本可加)仅作补充。
"""

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


def gray_transmittance(value):
    return (value, value, value)


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
        assert_rgb_close(self, res.transmittance, gray_transmittance(want_t))
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
        self.assertEqual(res.transmittance, (1.0, 1.0, 1.0))

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
        assert_rgb_close(self, res.transmittance, gray_transmittance(want_t))
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
        assert_rgb_close(self, pieces.transmittance, whole.transmittance, tol=1e-12)

    def test_per_channel_extinction_analytic_reference(self):
        """RGB 三个通道使用不同 sigma 时, 颜色和透过率逐通道积分。"""
        emission = (0.4, 0.8, 1.2)
        sigma = (0.2, 0.5, 1.0)
        z0, z1 = 0.5, 3.0
        length = z1 - z0
        res = composite([VolumeSegment(z0, z1, emission, sigma, id="rgb")])

        want_color = []
        want_transmittance = []
        for e_k, sigma_k in zip(emission, sigma):
            x = sigma_k * length
            want_color.append(e_k / sigma_k * (1.0 - math.exp(-x)))
            want_transmittance.append(math.exp(-x))
        assert_rgb_close(self, res.color, want_color)
        assert_rgb_close(self, res.transmittance, want_transmittance)
        entry = res.ledger[0]
        assert_rgb_close(self, entry.contribution, want_color)
        self.assertEqual(entry.transmittance_before, (1.0, 1.0, 1.0))
        assert_rgb_close(self, entry.transmittance_after, want_transmittance)

    def test_zero_absorption_preserves_emission_without_division(self):
        """单个或全部通道 sigma=0 时走长度积分, 不除零且发射保留。"""
        emission = (0.3, 0.6, 0.9)
        res = composite([VolumeSegment(0.0, 2.0, emission, (0.0, 0.5, 0.0))])
        x = 0.5 * 2.0
        want_color = (
            0.6,
            0.6 / 0.5 * (1.0 - math.exp(-x)),
            1.8,
        )
        assert_rgb_close(self, res.color, want_color)
        assert_rgb_close(self, res.transmittance, (1.0, math.exp(-x), 1.0))

        all_zero = composite([VolumeSegment(1.0, 4.0, emission, 0.0)])
        assert_rgb_close(self, all_zero.color, (0.9, 1.8, 2.7))
        self.assertEqual(all_zero.transmittance, (1.0, 1.0, 1.0))

    def test_scalar_sigma_matches_equal_channel_sigma(self):
        emission = (0.2, 0.7, 1.1)
        scalar = composite([VolumeSegment(0.0, 2.5, emission, 0.35)])
        channels = composite(
            [VolumeSegment(0.0, 2.5, emission, (0.35, 0.35, 0.35))]
        )
        assert_rgb_close(self, scalar.color, channels.color)
        assert_rgb_close(self, scalar.transmittance, channels.transmittance)

    def test_overlapping_colored_sigmas_add_per_channel(self):
        """重叠小区间的彩色消光按通道相加, 不使用平均吸收。"""
        items = [
            VolumeSegment(0.0, 1.0, (0.2, 0.2, 0.2), (0.1, 0.2, 0.3), id=1),
            VolumeSegment(0.0, 1.0, (0.1, 0.4, 0.0), (0.4, 0.0, 0.2), id=2),
        ]
        res = composite(items)
        sigma = (0.5, 0.2, 0.5)
        emission = (0.3, 0.6, 0.2)
        want_color = []
        want_transmittance = []
        for e_k, sigma_k in zip(emission, sigma):
            x = sigma_k
            want_color.append(e_k / sigma_k * (1.0 - math.exp(-x)))
            want_transmittance.append(math.exp(-x))
        assert_rgb_close(self, res.color, want_color)
        assert_rgb_close(self, res.transmittance, want_transmittance)

    def test_surface_applies_equally_to_colored_volume_transmittance(self):
        """表面 alpha 是几何遮挡, 同时作用于三个已有色的体透过率通道。"""
        volume = VolumeSegment(0.0, 1.0, (0.0, 0.0, 0.0), (0.2, 0.5, 0.8), id=1)
        surface = Surface(1.0, (0.0, 0.0, 0.0, 0.25), id=2)
        res = composite([volume, surface])
        assert_rgb_close(
            self,
            res.transmittance,
            tuple(0.75 * math.exp(-sigma_k) for sigma_k in (0.2, 0.5, 0.8)),
        )


class TestSurfaces(unittest.TestCase):
    def test_single_surface_premultiplied_over(self):
        # 预乘表面: C = c, T = 1 - a
        res = composite([Surface(1.0, (0.2, 0.4, 0.1, 0.5), id=0)])
        assert_rgb_close(self, res.color, (0.2, 0.4, 0.1))
        assert_rgb_close(self, res.transmittance, gray_transmittance(0.5))

    def test_same_depth_surfaces_sorted_by_id(self):
        """同 z 表面按 ID 升序合成(front-to-back)。"""
        # id=1 在前: c1=(0.5,0,0), a1=0.5; id=2 在后: c2=(0,0.5,0), a2=0.5
        s_front = Surface(1.0, (0.5, 0.0, 0.0, 0.5), id=1)
        s_back = Surface(1.0, (0.0, 0.5, 0.0, 0.5), id=2)
        res = composite([s_back, s_front])  # 输入顺序故意打乱
        # 手算: C = c1 + (1-a1)*c2 ; T = (1-a1)(1-a2)
        want = (0.5 + 0.5 * 0.0, 0.0 + 0.5 * 0.5, 0.0)
        assert_rgb_close(self, res.color, want)
        assert_rgb_close(self, res.transmittance, gray_transmittance(0.25))
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
        assert_rgb_close(self, res.transmittance, gray_transmittance(0.5))
        self.assertEqual([e.kind for e in res.ledger], ["volume", "surface", "volume"])

    def test_opaque_surface_hides_volume_starting_at_same_z(self):
        v1 = VolumeSegment(0.0, 1.0, (0.2, 0.0, 0.0), 0.0, id=1)
        opaque = Surface(1.0, (0.0, 0.3, 0.0, 1.0), id=2)
        v2 = VolumeSegment(1.0, 5.0, (0.0, 0.0, 9.0), 0.0, id=3)
        res = composite([v2, opaque, v1])
        # 不透明表面后透过率为 0, v2 无贡献
        assert_rgb_close(self, res.color, (0.2, 0.3, 0.0))
        self.assertEqual(res.transmittance, (0.0, 0.0, 0.0))

    def test_surface_then_colored_volume_combination(self):
        surface = Surface(1.0, (0.2, 0.1, 0.0, 0.5), id=1)
        volume = VolumeSegment(1.0, 2.0, (1.0, 1.0, 1.0), (0.0, 0.5, 1.0), id=2)
        res = composite([surface, volume])

        surface_t = 0.5
        volume_t = (1.0, math.exp(-0.5), math.exp(-1.0))
        volume_color = (1.0, 2.0 * (1.0 - math.exp(-0.5)), 1.0 - math.exp(-1.0))
        want_color = tuple(
            surface_color + surface_t * volume_color[k]
            for k, surface_color in enumerate((0.2, 0.1, 0.0))
        )
        assert_rgb_close(self, res.color, want_color)
        assert_rgb_close(
            self,
            res.transmittance,
            tuple(surface_t * volume_t[k] for k in range(3)),
        )


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
        assert_rgb_close(self, res.transmittance, gray_transmittance(0.5))
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
        self.assertEqual(res.ledger[0].transmittance_before, (1.0, 1.0, 1.0))
        self.assertEqual(res.ledger[-1].transmittance_after, res.transmittance)

    def test_empty_input(self):
        res = composite([])
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, (1.0, 1.0, 1.0))
        self.assertEqual(res.ledger, ())

    def test_apply_background(self):
        res = composite([Surface(1.0, (0.5, 0.0, 0.0, 0.5), id=0)])
        out = apply_background(res, (0.0, 0.0, 1.0))
        assert_rgb_close(self, out, (0.5, 0.0, 0.5))


class TestValidation(unittest.TestCase):
    def test_negative_sigma_rejected(self):
        with self.assertRaises(NegativeSigmaError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), -0.5)
        with self.assertRaises(NegativeSigmaError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), (0.2, -0.1, 0.3))

    def test_colored_sigma_shape_and_finiteness(self):
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), (0.2, 0.3))
        with self.assertRaises(ValidationError):
            VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), (0.2, float("nan"), 0.3))

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
        self.assertEqual(res.transmittance, (1.0, 1.0, 1.0))

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


if __name__ == "__main__":
    unittest.main()

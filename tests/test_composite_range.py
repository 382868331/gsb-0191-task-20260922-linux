"""composite_range / combine_results 单元测试。

参考值全部在测试内用 math.exp 按裁剪后的小区间独立手算, 不调用被测核心
生成期望值; "任意分窗重组 == 全量" 仅作为补充的往返一致性检查, 且全量
结果另有独立解析参考核对, 防止两个错误函数互相抵消。

手算场景(两段重叠介质 + 边界表面):
    A: 体段 [0, 6], e=(0.3, 0, 0),   sigma=0.2, id=1
    B: 体段 [2, 8], e=(0, 0.5, 0),   sigma=0.4, id=2
    S2: 表面 z=2, rgba=(0.1,0.1,0.1,0.5), id=3
    S6: 表面 z=6, rgba=(0.2,0,0,0.5),     id=4
分窗: W1=[0,2)  W2=[2,6)  W3=[6,10)
"""

import math
import unittest

from deep_composite import (
    CompositeResult,
    LedgerEntry,
    Surface,
    ValidationError,
    VolumeSegment,
    combine_results,
    composite,
    composite_range,
)

TOL = 1e-12


def rgb_close(tc, got, want, tol=TOL):
    tc.assertEqual(len(got), 3)
    for g, w in zip(got, want):
        tc.assertAlmostEqual(g, w, delta=tol)


def vol_c(e, sigma, length):
    """独立解析参考: 均匀介质一段的 (C, T)。"""
    x = sigma * length
    if x == 0.0:
        return tuple(ec * length for ec in e), 1.0
    c = tuple(ec / sigma * (1.0 - math.exp(-x)) for ec in e)
    return c, math.exp(-x)


def build_scene():
    return [
        VolumeSegment(0.0, 6.0, (0.3, 0.0, 0.0), 0.2, id=1),
        VolumeSegment(2.0, 8.0, (0.0, 0.5, 0.0), 0.4, id=2),
        Surface(2.0, (0.1, 0.1, 0.1, 0.5), id=3),
        Surface(6.0, (0.2, 0.0, 0.0, 0.5), id=4),
    ]


class TestWindowHandComputed(unittest.TestCase):
    """每个窗口的期望值独立手算(按截后长度重新积分)。"""

    def test_window_before_any_surface(self):
        # W1=[0,2): 仅 A, L=2, e=(.3,0,0), sigma=.2; 无表面(S2 在 z=2, 排除)
        res = composite_range(build_scene(), 0.0, 2.0)
        c, t = vol_c((0.3, 0.0, 0.0), 0.2, 2.0)
        rgb_close(self, res.color, c)
        self.assertAlmostEqual(res.transmittance, t, delta=TOL)
        self.assertEqual(len(res.ledger), 1)
        self.assertEqual(res.ledger[0].ids, (1,))
        self.assertEqual((res.ledger[0].z0, res.ledger[0].z1), (0.0, 2.0))

    def test_window_with_overlap_and_near_surface(self):
        # W2=[2,6): A、B 都跨边界, 裁剪后均为 [2,6], 合并 e=(.3,.5,0), sigma=.6
        # 表面 S2 恰在 near(计入), S6 恰在 far(排除)。
        # 顺序: S2 -> 体段 [2,6]
        res = composite_range(build_scene(), 2.0, 6.0)
        cv, tv = vol_c((0.3, 0.5, 0.0), 0.6, 4.0)
        want_c = tuple(0.1 + 0.5 * cv[i] for i in range(3))
        want_t = 0.5 * tv
        rgb_close(self, res.color, want_c)
        self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)
        kinds = [e.kind for e in res.ledger]
        self.assertEqual(kinds, ["surface", "volume"])
        # 账本保留原样本 ID 与裁剪范围
        self.assertEqual(res.ledger[0].ids, (3,))
        self.assertEqual(res.ledger[1].ids, (1, 2))
        self.assertEqual((res.ledger[1].z0, res.ledger[1].z1), (2.0, 6.0))
        # 透过率只来自截后长度 4: 不含窗外光学厚度
        self.assertAlmostEqual(tv, math.exp(-2.4), delta=TOL)

    def test_window_far_side(self):
        # W3=[6,10): A 完全在外; B 裁剪为 [6,8], e=(0,.5,0), sigma=.4, L=2
        # S6 恰在 near(计入); z=10 处无样本。
        res = composite_range(build_scene(), 6.0, 10.0)
        cv, tv = vol_c((0.0, 0.5, 0.0), 0.4, 2.0)
        want_c = (0.2 + 0.5 * cv[0], 0.5 * cv[1], 0.5 * cv[2])
        want_t = 0.5 * tv
        rgb_close(self, res.color, want_c)
        self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)
        self.assertEqual([e.kind for e in res.ledger], ["surface", "volume"])
        self.assertEqual(res.ledger[1].ids, (2,))
        self.assertEqual((res.ledger[1].z0, res.ledger[1].z1), (6.0, 8.0))


class TestBoundarySemantics(unittest.TestCase):
    def test_surface_exactly_at_near_included(self):
        items = [Surface(3.0, (0.7, 0.0, 0.0, 0.5), id="s")]
        res = composite_range(items, 3.0, 5.0)
        rgb_close(self, res.color, (0.7, 0.0, 0.0))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)
        self.assertEqual(len(res.ledger), 1)

    def test_surface_exactly_at_far_excluded(self):
        items = [Surface(5.0, (0.7, 0.0, 0.0, 0.5), id="s")]
        res = composite_range(items, 3.0, 5.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_volume_touching_far_with_zero_length_is_empty(self):
        # 体段恰在 far 结束、且从 far 开始的部分长度为 0: 空
        items = [VolumeSegment(5.0, 6.0, (1.0, 1.0, 1.0), 0.3)]
        res = composite_range(items, 5.0, 5.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_degenerate_window_is_empty_not_error(self):
        res = composite_range(build_scene(), 4.0, 4.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_window_with_no_content(self):
        res = composite_range(build_scene(), 8.5, 9.5)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_reversed_window_rejected(self):
        with self.assertRaises(ValidationError):
            composite_range(build_scene(), 6.0, 2.0)

    def test_bad_bounds_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError, msg=f"z_near={bad}"):
                composite_range(build_scene(), bad, 5.0)
            with self.assertRaises(ValidationError, msg=f"z_far={bad}"):
                composite_range(build_scene(), 0.0, bad)

    def test_window_independent_of_pre_window_occlusion(self):
        """窗前的不透明表面不得计入窗口结果(独立窗口, 非"从相机所见")。"""
        scene = build_scene() + [Surface(1.0, (0.9, 0.9, 0.9, 1.0), id=99)]
        plain = composite_range(build_scene(), 2.0, 6.0)
        blocked = composite_range(scene, 2.0, 6.0)
        rgb_close(self, blocked.color, plain.color)
        self.assertEqual(blocked.transmittance, plain.transmittance)
        self.assertFalse(any(99 in e.ids for e in blocked.ledger))


class TestRecomposition(unittest.TestCase):
    """相邻窗口按 C=C1+T1*C2, T=T1*T2 拼回; 并与独立解析全量参考核对。"""

    def test_combine_formula_hand_reference(self):
        # 直接用手算的两个窗口结果验证 combine_results 代数
        r1 = composite_range(build_scene(), 0.0, 2.0)
        r2 = composite_range(build_scene(), 2.0, 6.0)
        joined = combine_results(r1, r2)
        want_c = tuple(r1.color[i] + r1.transmittance * r2.color[i] for i in range(3))
        want_t = r1.transmittance * r2.transmittance
        rgb_close(self, joined.color, want_c)
        self.assertAlmostEqual(joined.transmittance, want_t, delta=TOL)
        # 账本链接续、可加
        self.assertEqual(
            joined.ledger[0].transmittance_before, 1.0
        )
        for prev, cur in zip(joined.ledger, joined.ledger[1:]):
            self.assertAlmostEqual(
                prev.transmittance_after, cur.transmittance_before, delta=1e-14
            )
        sums = [0.0, 0.0, 0.0]
        for e in joined.ledger:
            for i in range(3):
                sums[i] += e.contribution[i]
        rgb_close(self, sums, joined.color, tol=1e-13)

    def test_three_windows_reassemble_to_full(self):
        """任意分窗(含非端点切分)重组 == composite() 全量。"""
        scene = build_scene()
        windows = [
            composite_range(scene, 0.0, 1.37),
            composite_range(scene, 1.37, 4.8),
            composite_range(scene, 4.8, 10.0),
        ]
        joined = windows[0]
        for w in windows[1:]:
            joined = combine_results(joined, w)
        full = composite(scene)
        rgb_close(self, joined.color, full.color, tol=1e-12)
        self.assertAlmostEqual(joined.transmittance, full.transmittance, delta=1e-12)

    def test_full_scene_independent_analytic_reference(self):
        """全量结果独立手算: 不依赖 composite_range/combine_results。

        区间划分: [0,2] 仅A; z=2 表面S2; [2,6] A+B; z=6 表面S6; [6,8] 仅B。
        """
        c1, t1 = vol_c((0.3, 0.0, 0.0), 0.2, 2.0)
        c2, t2 = vol_c((0.3, 0.5, 0.0), 0.6, 4.0)
        c3, t3 = vol_c((0.0, 0.5, 0.0), 0.4, 2.0)
        s2_c, s2_a = (0.1, 0.1, 0.1), 0.5
        s6_c, s6_a = (0.2, 0.0, 0.0), 0.5

        T = t1
        acc = list(c1)
        acc = [acc[i] + T * s2_c[i] for i in range(3)]
        T *= 1.0 - s2_a
        acc = [acc[i] + T * c2[i] for i in range(3)]
        T *= t2
        acc = [acc[i] + T * s6_c[i] for i in range(3)]
        T *= 1.0 - s6_a
        acc = [acc[i] + T * c3[i] for i in range(3)]
        T *= t3

        full = composite(build_scene())
        rgb_close(self, full.color, tuple(acc))
        self.assertAlmostEqual(full.transmittance, T, delta=TOL)

        # 同一参考同时核对三窗重组, 避免错误互相抵消
        w1 = composite_range(build_scene(), 0.0, 2.0)
        w2 = composite_range(build_scene(), 2.0, 6.0)
        w3 = composite_range(build_scene(), 6.0, 10.0)
        joined = combine_results(combine_results(w1, w2), w3)
        rgb_close(self, joined.color, tuple(acc))
        self.assertAlmostEqual(joined.transmittance, T, delta=TOL)

    def test_partition_covers_all_surface_boundaries(self):
        """表面恰在分窗点时: 归近窗(含)而不归远窗, 不重不漏。"""
        scene = build_scene()
        # 切点恰好取在两个表面深度上
        w1 = composite_range(scene, 0.0, 2.0)   # 不含 S2
        w2 = composite_range(scene, 2.0, 6.0)   # 含 S2, 不含 S6
        w3 = composite_range(scene, 6.0, 8.0)   # 含 S6
        joined = combine_results(combine_results(w1, w2), w3)
        full = composite(scene)
        rgb_close(self, joined.color, full.color)
        self.assertAlmostEqual(joined.transmittance, full.transmittance, delta=TOL)
        # S2 只在 W2, S6 只在 W3
        self.assertFalse(any(3 in e.ids for e in w1.ledger))
        self.assertTrue(any(3 in e.ids for e in w2.ledger))
        self.assertFalse(any(4 in e.ids for e in w2.ledger))
        self.assertTrue(any(4 in e.ids for e in w3.ledger))


class TestRangeValidationAndTypes(unittest.TestCase):
    def test_item_limit_applies(self):
        items = [Surface(float(i), (0.1, 0.1, 0.1, 0.1), id=i) for i in range(101)]
        with self.assertRaises(Exception):
            composite_range(items, 0.0, 1000.0)

    def test_wrong_item_type(self):
        with self.assertRaises(ValidationError):
            composite_range([("x",)], 0.0, 1.0)

    def test_combine_rejects_non_result(self):
        good = composite_range([], 0.0, 1.0)
        with self.assertRaises(ValidationError):
            combine_results(good, (1, 2, 3))

    def test_result_is_composite_result(self):
        res = composite_range(build_scene(), 0.0, 2.0)
        self.assertIsInstance(res, CompositeResult)
        self.assertIsInstance(res.ledger[0], LedgerEntry)


if __name__ == "__main__":
    unittest.main()

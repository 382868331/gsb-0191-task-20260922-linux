"""composite_range 深度窗合成与 combine_results 拼回的单元测试。

核验方式(三者独立, 防止"两个错误函数互相抵消"):
  1. 手算: 两段重叠介质 + 恰好落在 near/far 边界上的表面, 逐步写出
     每一步的解析积分与透过率(仅用 math.exp, 不调用被测核心);
  2. 独立分段解析积分参考 reference_range(): 在测试内另写的一份直接
     扫描实现, 与被测核心结构不同, 逐窗核对;
  3. 重组不变量: 任意切点(表面深度、体段端点、任意分数位置)分窗后
     按 C=C1+T1*C2、T=T1*T2 拼回, 同时等于全量 composite() 与参考。
"""

import math
import unittest

from deep_composite import (
    MAX_ITEMS,
    CompositeResult,
    ItemLimitError,
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


def _id_key(item_id):
    return (0, item_id) if isinstance(item_id, int) else (1, item_id)


def _analytic_segment(e, sigma, length):
    """独立解析积分(测试自有, 不经过 deep_composite 内部)。"""
    if sigma == 0.0:
        return (e[0] * length, e[1] * length, e[2] * length), 1.0
    t = math.exp(-sigma * length)
    return (
        e[0] / sigma * (1.0 - t),
        e[1] / sigma * (1.0 - t),
        e[2] / sigma * (1.0 - t),
    ), t


def reference_range(items, z_near, z_far):
    """独立分段解析积分参考: 直接裁剪 + 端点扫描, 返回 (color, T)。

    与被测实现分别书写(事件组织方式不同), 仅依赖 math.exp。
    """
    vols = []
    surfs = []
    for it in items:
        if isinstance(it, Surface):
            if z_near <= it.z < z_far:
                surfs.append(it)
        else:
            a = max(it.z0, z_near)
            b = min(it.z1, z_far)
            if b > a:
                vols.append((a, b, it.emission, it.sigma))
    surfs.sort(key=lambda s: (s.z, _id_key(s.id)))

    cuts = set()
    for a, b, _e, _s in vols:
        cuts.add(a)
        cuts.add(b)
    for s in surfs:
        cuts.add(s.z)
    ordered = sorted(cuts)

    color = [0.0, 0.0, 0.0]
    trans = 1.0
    prev = None
    for z in ordered:
        if prev is not None and z > prev:
            sig = 0.0
            emi = [0.0, 0.0, 0.0]
            for a, b, e, s in vols:
                if a <= prev and b >= z:
                    sig += s
                    emi[0] += e[0]
                    emi[1] += e[1]
                    emi[2] += e[2]
            c_seg, t_seg = _analytic_segment((emi[0], emi[1], emi[2]), sig, z - prev)
            color[0] += trans * c_seg[0]
            color[1] += trans * c_seg[1]
            color[2] += trans * c_seg[2]
            trans *= t_seg
        for s in (x for x in surfs if x.z == z):
            color[0] += trans * s.rgba[0]
            color[1] += trans * s.rgba[1]
            color[2] += trans * s.rgba[2]
            trans *= 1.0 - s.rgba[3]
        prev = z
    return (color[0], color[1], color[2]), trans


def hand_scene():
    """两段重叠介质 [0,4] 与 [2,6], 边界表面 z=2/z=4, 内部表面 z=3。"""
    return [
        VolumeSegment(0.0, 4.0, (0.30, 0.10, 0.02), 0.40, id=10),
        VolumeSegment(2.0, 6.0, (0.05, 0.20, 0.35), 0.25, id=11),
        Surface(2.0, (0.40, 0.20, 0.10, 0.50), id=21),  # 恰在 near
        Surface(4.0, (0.10, 0.10, 0.10, 0.50), id=22),  # 恰在 far
        Surface(3.0, (0.20, 0.00, 0.00, 0.25), id=23),  # 窗内
    ]


class TestRangeHandComputed(unittest.TestCase):
    """手算: 窗 [2,4), 表面 near 计入/far 排除, 跨界体段按截长重积分。"""

    def test_window_two_to_four_step_by_step(self):
        res = composite_range(hand_scene(), 2.0, 4.0)

        # 窗内: S1(z=2) 先于从 z=2 开始(裁剪后)的体段;
        # [2,3) 两段完全重叠: sigma=0.65, e=(0.35,0.30,0.37);
        # S3(z=3); [3,4) 同一重叠介质; S2(z=4) 被排除。
        e = (0.35, 0.30, 0.37)
        sig = 0.65
        c_seg, t_seg = _analytic_segment(e, sig, 1.0)

        # 第一步: S1
        c = [0.40, 0.20, 0.10]
        t = 1.0 - 0.50  # 0.5
        # 第二步: 体段 [2,3)
        c = [c[k] + t * c_seg[k] for k in range(3)]
        t = t * t_seg
        # 第三步: S3
        c[0] += t * 0.20
        t = t * (1.0 - 0.25)
        # 第四步: 体段 [3,4)
        c = [c[k] + t * c_seg[k] for k in range(3)]
        t = t * t_seg

        rgb_close(self, res.color, (c[0], c[1], c[2]))
        self.assertAlmostEqual(res.transmittance, t, delta=TOL)

        # 账本顺序: 表面(near) -> 体段 -> 表面 -> 体段; far 表面不在
        kinds = [e.kind for e in res.ledger]
        self.assertEqual(kinds, ["surface", "volume", "surface", "volume"])
        ids_seq = [e.ids for e in res.ledger]
        self.assertEqual(ids_seq[0], (21,))
        self.assertEqual(ids_seq[2], (23,))
        self.assertFalse(any(22 in e.ids for e in res.ledger))
        # 两个重叠小区间都保留了原始样本 ID, 区间被裁进窗内
        self.assertEqual(set(res.ledger[1].ids), {10, 11})
        self.assertEqual((res.ledger[1].z0, res.ledger[1].z1), (2.0, 3.0))
        self.assertEqual(set(res.ledger[3].ids), {10, 11})
        self.assertEqual((res.ledger[3].z0, res.ledger[3].z1), (3.0, 4.0))

    def test_matches_independent_reference(self):
        # 手算场景之外再用独立参考核对, 避免两份实现共错相消
        items = hand_scene()
        for a, b in [(2.0, 4.0), (0.0, 6.0), (1.0, 3.5), (2.0, 2.5), (-3.0, 9.0)]:
            res = composite_range(items, a, b)
            want_c, want_t = reference_range(items, a, b)
            rgb_close(self, res.color, want_c)
            self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)


class TestRangeSurfaceBoundaries(unittest.TestCase):
    def test_surface_exactly_at_near_included(self):
        items = [Surface(2.0, (0.4, 0.0, 0.0, 0.5), id=1)]
        res = composite_range(items, 2.0, 4.0)
        rgb_close(self, res.color, (0.4, 0.0, 0.0))
        self.assertAlmostEqual(res.transmittance, 0.5, delta=TOL)
        self.assertEqual(len(res.ledger), 1)

    def test_surface_exactly_at_far_excluded(self):
        items = [Surface(4.0, (0.4, 0.0, 0.0, 0.5), id=1)]
        res = composite_range(items, 2.0, 4.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_surface_at_near_before_volume_starting_at_near(self):
        # 体段 [0,10] 裁到窗 [2,3) 后从 near 开始; near 处表面仍在体段之前。
        items = [
            VolumeSegment(0.0, 10.0, (0.0, 1.0, 0.0), 1.0, id=1),
            Surface(2.0, (0.0, 0.0, 1.0, 0.5), id=2),
        ]
        res = composite_range(items, 2.0, 3.0)
        c_seg, t_seg = _analytic_segment((0.0, 1.0, 0.0), 1.0, 1.0)
        # 表面先: C=(0,0,1), T=0.5; 再体段
        want = (0.0, 0.5 * c_seg[1], 1.0)
        want_t = 0.5 * t_seg
        rgb_close(self, res.color, want)
        self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)
        self.assertEqual([e.kind for e in res.ledger], ["surface", "volume"])


class TestRangeVolumeClipping(unittest.TestCase):
    def test_crossing_segment_reintegrated_by_clipped_length(self):
        # 单段 [0,10] 跨两侧边界, 窗 [4,6): 截长 2, 必须按 2 重新积分
        e = (1.0, 2.0, 3.0)
        items = [VolumeSegment(0.0, 10.0, e, 0.3, id=7)]
        res = composite_range(items, 4.0, 6.0)
        want_c, want_t = _analytic_segment(e, 0.3, 2.0)
        rgb_close(self, res.color, want_c)
        self.assertAlmostEqual(res.transmittance, want_t, delta=TOL)
        self.assertEqual((res.ledger[0].z0, res.ledger[0].z1), (4.0, 6.0))
        self.assertEqual(res.ledger[0].ids, (7,))
        # 防呆: 绝不能误用截前长度 10
        wrong_c, _ = _analytic_segment(e, 0.3, 10.0)
        self.assertGreater(abs(res.color[0] - wrong_c[0]), 1e-3)

    def test_zero_sigma_clipped_length(self):
        e = (0.1, 0.0, 2.0)
        res = composite_range([VolumeSegment(0.0, 10.0, e, 0.0, id=1)], 2.0, 5.0)
        rgb_close(self, res.color, (0.3, 0.0, 6.0))  # e * 3
        self.assertEqual(res.transmittance, 1.0)

    def test_segments_only_touching_boundaries_have_no_interior(self):
        # 体段在窗前结束(端点 z=2)或恰在 far 才开始: 半开窗内无长度
        items = [
            VolumeSegment(0.0, 2.0, (1.0, 1.0, 1.0), 0.5, id=1),
            VolumeSegment(4.0, 8.0, (1.0, 1.0, 1.0), 0.5, id=2),
        ]
        res = composite_range(items, 2.0, 4.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_ledger_keeps_original_id_and_clipped_span_per_subinterval(self):
        items = [
            VolumeSegment(0.0, 10.0, (0.2, 0.0, 0.0), 0.1, id="A"),
            VolumeSegment(3.0, 7.0, (0.0, 0.2, 0.0), 0.2, id="B"),
            Surface(5.0, (0.0, 0.0, 0.3, 0.5), id="S"),
        ]
        res = composite_range(items, 2.0, 8.0)
        vol_entries = [e for e in res.ledger if e.kind == "volume"]
        spans = [(e.z0, e.z1, e.ids) for e in vol_entries]
        # 端点 2,3,5,7,8(表面在 5 把区间切开)
        self.assertEqual(spans[0], (2.0, 3.0, ("A",)))
        self.assertEqual(spans[1], (3.0, 5.0, ("A", "B")))
        self.assertEqual(spans[2], (5.0, 7.0, ("A", "B")))
        self.assertEqual(spans[3], (7.0, 8.0, ("A",)))
        self.assertEqual([e.ids for e in res.ledger if e.kind == "surface"], [("S",)])


class TestRangeIndependentOfFrontOcclusion(unittest.TestCase):
    """窗前遮挡绝不进入独立窗口结果。"""

    def test_opaque_surface_before_window_has_no_effect(self):
        items = [
            VolumeSegment(0.0, 10.0, (0.0, 0.5, 0.0), 0.3, id=1),
            Surface(0.5, (1.0, 1.0, 1.0, 1.0), id=99),  # 窗前不透明遮挡
            Surface(3.0, (0.2, 0.0, 0.0, 0.5), id=2),
        ]
        plain = composite_range(items[::1], 2.0, 4.0)
        # 去掉窗前遮挡, 窗结果必须逐位一致
        without = composite_range([items[0], items[2]], 2.0, 4.0)
        self.assertEqual(plain.color, without.color)
        self.assertEqual(plain.transmittance, without.transmittance)
        self.assertEqual(plain.ledger, without.ledger)

        # 与"窗前没有任何东西"的独立参考一致(透过率从 1 起算)
        want_c, want_t = reference_range([items[0], items[2]], 2.0, 4.0)
        rgb_close(self, plain.color, want_c)
        self.assertAlmostEqual(plain.transmittance, want_t, delta=TOL)
        # 窗前不透明表面没有被记入账本
        self.assertFalse(any(99 in e.ids for e in plain.ledger))


class TestEmptyAndReversedWindow(unittest.TestCase):
    def test_empty_window_is_zero_color_and_unit_transmittance(self):
        items = hand_scene()
        res = composite_range(items, 7.0, 9.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())

    def test_zero_width_window_is_empty(self):
        res = composite_range(hand_scene(), 3.0, 3.0)
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, 1.0)
        self.assertEqual(res.ledger, ())
        # 恰在该点的表面也因半开 [z,z) 被排除
        res2 = composite_range([Surface(3.0, (0.5, 0.5, 0.5, 0.5))], 3.0, 3.0)
        self.assertEqual(res2.ledger, ())

    def test_reversed_window_rejected(self):
        with self.assertRaises(ValidationError):
            composite_range(hand_scene(), 4.0, 2.0)

    def test_bad_bounds_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError):
                composite_range([], bad, 1.0)
            with self.assertRaises(ValidationError):
                composite_range([], 0.0, bad)
        with self.assertRaises(ValidationError):
            composite_range([], True, 1.0)
        with self.assertRaises(ValidationError):
            composite_range([], 0.0, False)

    def test_item_limit_and_types_apply(self):
        items = [Surface(float(i), (0.1, 0.1, 0.1, 0.1), id=i) for i in range(MAX_ITEMS + 1)]
        with self.assertRaises(ItemLimitError):
            composite_range(items, 0.0, 1000.0)
        with self.assertRaises(ValidationError):
            composite_range([("nope",)], 0.0, 1.0)
        with self.assertRaises(ValidationError):
            composite_range(42, 0.0, 1.0)


class TestCombineResults(unittest.TestCase):
    def test_hand_computed_two_windows(self):
        # 两个表面分别落在相邻窗: 手算 C=C1+T1*C2, T=T1*T2
        items = [
            Surface(1.0, (0.5, 0.0, 0.0, 0.5), id=1),
            Surface(2.0, (0.0, 0.5, 0.0, 0.5), id=2),
        ]
        r1 = composite_range(items, 0.0, 1.5)
        r2 = composite_range(items, 1.5, 3.0)
        joined = combine_results(r1, r2)
        rgb_close(self, joined.color, (0.5, 0.5 * 0.5, 0.0))
        self.assertAlmostEqual(joined.transmittance, 0.25, delta=TOL)
        # 与全量合成一致
        full = composite(items)
        rgb_close(self, joined.color, full.color)
        self.assertAlmostEqual(joined.transmittance, full.transmittance, delta=TOL)

    def test_combined_ledger_chain_continuous_and_additive(self):
        items = hand_scene()
        r1 = composite_range(items, 0.0, 3.0)
        r2 = composite_range(items, 3.0, 6.0)
        joined = combine_results(r1, r2)
        n1 = len(r1.ledger)
        # 接缝连续: 前段末条 after == 后段首条(全局化后)before == T1
        self.assertAlmostEqual(
            joined.ledger[n1 - 1].transmittance_after,
            joined.ledger[n1].transmittance_before,
            delta=1e-15,
        )
        self.assertAlmostEqual(
            joined.ledger[n1].transmittance_before,
            r1.transmittance,
            delta=1e-15,
        )
        for prev, cur in zip(joined.ledger, joined.ledger[1:]):
            self.assertAlmostEqual(prev.transmittance_after, cur.transmittance_before, delta=1e-15)
        self.assertEqual(joined.ledger[0].transmittance_before, 1.0)
        self.assertAlmostEqual(joined.ledger[-1].transmittance_after, joined.transmittance, delta=1e-15)
        # 可加性精确
        sums = [0.0, 0.0, 0.0]
        for e in joined.ledger:
            for k in range(3):
                sums[k] += e.contribution[k]
        self.assertEqual(tuple(sums), joined.color)
        # 后段原始 ID 与裁剪范围保留
        tail_ids = {iid for e in joined.ledger[len(r1.ledger):] for iid in e.ids}
        self.assertIn(11, tail_ids)

    def test_combine_type_check(self):
        good = composite_range([], 0.0, 1.0)
        with self.assertRaises(ValidationError):
            combine_results(good, (1.0, 2.0))
        with self.assertRaises(ValidationError):
            combine_results("x", good)

    def test_three_window_chain(self):
        items = hand_scene()
        r1 = composite_range(items, -1.0, 2.0)
        r2 = composite_range(items, 2.0, 4.0)
        r3 = composite_range(items, 4.0, 9.0)
        joined = combine_results(combine_results(r1, r2), r3)
        full = composite(items)
        rgb_close(self, joined.color, full.color)
        self.assertAlmostEqual(joined.transmittance, full.transmittance, delta=TOL)
        # 颜色恒等式直接核对
        c12 = tuple(r1.color[k] + r1.transmittance * r2.color[k] for k in range(3))
        t12 = r1.transmittance * r2.transmittance
        want_c = tuple(c12[k] + t12 * r3.color[k] for k in range(3))
        want_t = t12 * r3.transmittance
        rgb_close(self, joined.color, want_c, tol=1e-11)
        self.assertAlmostEqual(joined.transmittance, want_t, delta=1e-12)


class TestArbitraryPartitions(unittest.TestCase):
    """任意分窗再组合 == 全量 composite() == 独立解析参考。"""

    SCENES = [
        hand_scene(),
        [  # 含间隙、零消光、多表面、str ID
            VolumeSegment(0.0, 2.0, (0.3, 0.1, 0.0), 0.4, id=1),
            VolumeSegment(1.0, 5.0, (0.0, 0.2, 0.5), 0.0, id=2),
            VolumeSegment(4.0, 8.0, (0.2, 0.2, 0.2), 0.9, id=3),
            Surface(1.0, (0.2, 0.2, 0.2, 0.5), id=10),
            Surface(4.0, (0.1, 0.0, 0.0, 0.3), id="s"),
            Surface(6.5, (0.0, 0.0, 0.2, 0.8), id=11),
        ],
    ]

    CUT_SETS = [
        [2.0, 3.0, 4.0, 6.0],                       # 体段端点与边界表面
        [1.5, 2.5, 3.7, 5.2, 6.9],                  # 任意分数位置
        [1.0, 4.0, 6.5],                            # 表面深度
        [0.7, 2.0, 4.0, 6.5, 7.3],                  # 混合
        [3.0],                                       # 单刀
        [],                                          # 不分(整窗)
    ]

    LO, HI = -10.0, 20.0

    def _partition_and_join(self, items, cuts):
        bounds = [self.LO, *cuts, self.HI]
        results = [
            composite_range(items, bounds[i], bounds[i + 1])
            for i in range(len(bounds) - 1)
        ]
        joined = results[0]
        for nxt in results[1:]:
            joined = combine_results(joined, nxt)
        return joined

    def test_all_partitions_match_full_and_reference(self):
        for si, items in enumerate(self.SCENES):
            full = composite(items)
            ref_c, ref_t = reference_range(items, self.LO, self.HI)
            rgb_close(self, full.color, ref_c)
            self.assertAlmostEqual(full.transmittance, ref_t, delta=TOL)
            for ci, cuts in enumerate(self.CUT_SETS):
                with self.subTest(scene=si, cuts=cuts):
                    joined = self._partition_and_join(items, cuts)
                    rgb_close(self, joined.color, full.color, tol=1e-11)
                    self.assertAlmostEqual(joined.transmittance, full.transmittance, delta=1e-11)
                    rgb_close(self, joined.color, ref_c, tol=1e-11)
                    self.assertAlmostEqual(joined.transmittance, ref_t, delta=1e-11)
                    # 拼回账本仍可加
                    sums = [0.0, 0.0, 0.0]
                    for e in joined.ledger:
                        for k in range(3):
                            sums[k] += e.contribution[k]
                    for k in range(3):
                        self.assertAlmostEqual(sums[k], joined.color[k], delta=1e-12)

    def test_partition_includes_empty_middle_window(self):
        # 切点落在无样本间隙: 中间空窗为零色 T=1, 拼回后结果不变
        items = [
            Surface(1.0, (0.5, 0.0, 0.0, 0.5)),
            Surface(9.0, (0.0, 0.5, 0.0, 0.5)),
        ]
        joined = self._partition_and_join(items, [3.0, 7.0])
        full = composite(items)
        rgb_close(self, joined.color, full.color)
        self.assertAlmostEqual(joined.transmittance, full.transmittance, delta=TOL)


if __name__ == "__main__":
    unittest.main()

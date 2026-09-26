"""simplify_surfaces 单元测试。

最优性用独立暴力枚举核对: 枚举全部合法分组(深度区间切分 + 每组压缩/
保留两种形态), 直接比较输出数量与平局时的分组终点序列, 不调用被测核心
得到期望值。
"""

import itertools
import unittest

from deep_composite import (
    ItemLimitError,
    MAX_ITEMS,
    Surface,
    ValidationError,
    VolumeSegment,
    composite,
    simplify_surfaces,
)


def _id_key(s):
    return (s.z, (0, s.id) if isinstance(s.id, int) else (1, s.id))


def fold(seg):
    """独立重写折叠公式(预乘 front-to-back)。"""
    acc = [0.0, 0.0, 0.0]
    rem = 1.0
    for s in seg:
        r, g, b, a = s.rgba
        for i, c in enumerate((r, g, b)):
            acc[i] += rem * c
        rem *= 1.0 - a
    return tuple(acc), 1.0 - rem


def states_at(seq, target_zs):
    """独立重写前缀状态求值: 每个目标深度边界之后的 (color, T)。"""
    acc = [0.0, 0.0, 0.0]
    t = 1.0
    seq = sorted(seq, key=_id_key)
    out = {}
    i = 0
    for z in target_zs:
        while i < len(seq) and seq[i].z <= z:
            r, g, b, a = seq[i].rgba
            for j, c in enumerate((r, g, b)):
                acc[j] += t * c
            t *= 1.0 - a
            i += 1
        out[z] = (acc[0], acc[1], acc[2], t)
    return out


def brute_force_best(surfaces, tol):
    """暴力枚举全部合法分组, 返回 ((最少数量, 终点序列), 分组方案)。

    分组方案: tuple of (kind, p, q), kind "c" 压缩 / "k" 保留, p/q 为深度下标。
    """
    seq = sorted(surfaces, key=_id_key)
    depths = []
    for s in seq:
        if not depths or depths[-1][0] != s.z:
            depths.append([s.z, [s]])
        else:
            depths[-1][1].append(s)
    nd = len(depths)

    def t_enter(p):
        t = 1.0
        for d in range(p):
            for s in depths[d][1]:
                t *= 1.0 - s.rgba[3]
        return t

    def compress_ok(p, q):
        seg = [s for d in range(p, q + 1) for s in depths[d][1]]
        color, alpha = fold(seg)
        if not all(0.0 <= c <= 1.0 for c in color + (alpha,)):
            return False
        te = t_enter(p)
        acc = [0.0, 0.0, 0.0]
        rem = 1.0
        for d in range(p, q):  # 区间内部边界(不含终点)
            for s in depths[d][1]:
                r, g, b, a = s.rgba
                for j, c in enumerate((r, g, b)):
                    acc[j] += rem * c
                rem *= 1.0 - a
            if te * max(acc) > tol or te * (1.0 - rem) > tol:
                return False
        return True

    best = None
    for cut_mask in itertools.product((0, 1), repeat=nd - 1):
        spans = []
        start = 0
        for i, cut in enumerate(cut_mask):
            if cut:
                spans.append((start, i))
                start = i + 1
        spans.append((start, nd - 1))
        options = []
        for p, q in spans:
            opts = []
            if compress_ok(p, q):
                opts.append(("c", p, q))
            if p == q:
                opts.append(("k", p, q))
            options.append(opts)
        for combo in itertools.product(*options):
            count = sum(1 if k == "c" else len(depths[p][1]) for k, p, q in combo)
            ends = tuple(depths[q][0] for _, _, q in combo)
            key = (count, ends)
            if best is None or key < best[0]:
                best = (key, combo)
    return best


class TestBruteForceOptimality(unittest.TestCase):
    """小序列: DP 结果须与暴力枚举的最优值一致。"""

    def check_against_brute(self, surfaces, tol):
        res = simplify_surfaces(surfaces, tolerance=tol)
        (want_count, want_ends), _combo = brute_force_best(surfaces, tol)
        self.assertEqual(res.output_count, want_count)
        self.assertEqual(res.group_ends, want_ends)
        self.assertLessEqual(res.worst_error, tol + 1e-15)
        return res

    def test_small_sequences(self):
        cases = [
            [Surface(1.0, (0.1, 0.2, 0.3, 0.2), id=1),
             Surface(2.0, (0.2, 0.1, 0.1, 0.3), id=2),
             Surface(3.0, (0.1, 0.1, 0.4, 0.1), id=3)],
            [Surface(1.0, (0.3, 0.1, 0.0, 0.4), id=1),
             Surface(1.0, (0.1, 0.3, 0.2, 0.2), id=2),
             Surface(2.0, (0.2, 0.2, 0.2, 0.5), id=3),
             Surface(4.0, (0.0, 0.1, 0.3, 0.1), id=4)],
            [Surface(0.5, (0.05, 0.05, 0.05, 0.05), id=i) for i in range(6)],
            [Surface(1.0, (0.4, 0.4, 0.4, 0.9), id=1),
             Surface(2.0, (0.01, 0.01, 0.01, 0.02), id=2),
             Surface(3.0, (0.01, 0.02, 0.01, 0.03), id=3),
             Surface(4.0, (0.3, 0.3, 0.3, 0.8), id=4)],
        ]
        for surfaces in cases:
            for tol in (0.0, 0.05, 0.2, 1.0):
                with self.subTest(tol=tol, n=len(surfaces)):
                    self.check_against_brute(surfaces, tol)

    def test_same_depth_surfaces_fold_to_one(self):
        surfaces = [
            Surface(2.0, (0.2, 0.1, 0.0, 0.5), id=2),
            Surface(2.0, (0.1, 0.3, 0.1, 0.25), id=1),
            Surface(2.0, (0.0, 0.1, 0.4, 0.1), id=3),
        ]
        res = simplify_surfaces(surfaces, tolerance=0.0)
        self.assertEqual(res.output_count, 1)
        only = res.surfaces[0]
        self.assertEqual(only.z, 2.0)
        want_c, want_a = fold(sorted(surfaces, key=lambda s: s.id))
        for got, want in zip(only.rgba, want_c + (want_a,)):
            self.assertAlmostEqual(got, want, delta=1e-15)

    def test_tolerance_zero_blocks_cross_depth_merge(self):
        surfaces = [
            Surface(1.0, (0.2, 0.2, 0.2, 0.3), id=1),
            Surface(2.0, (0.3, 0.1, 0.1, 0.4), id=2),
        ]
        res = simplify_surfaces(surfaces, tolerance=0.0)
        self.assertEqual(res.output_count, 2)
        self.assertEqual(res.worst_error, 0.0)

    def test_transparent_layers_around_near_opaque(self):
        surfaces = [
            Surface(1.0, (0.001, 0.001, 0.001, 0.001), id=1),
            Surface(2.0, (0.002, 0.001, 0.001, 0.002), id=2),
            Surface(3.0, (0.9, 0.9, 0.9, 0.999), id=3),
            Surface(4.0, (0.001, 0.002, 0.001, 0.001), id=4),
            Surface(5.0, (0.001, 0.001, 0.002, 0.002), id=5),
        ]
        tight = simplify_surfaces(surfaces, tolerance=1e-6)
        self.assertEqual(tight.output_count, 5)
        loose = simplify_surfaces(surfaces, tolerance=0.05)
        # 最优: [透明层..不透明层] 压到不透明层深度(内部边界都在不透明层
        # 之前, 偏差可忽略), 不透明层之后的透明层再压成一个。
        self.assertEqual(loose.output_count, 2)
        self.assertLessEqual(loose.worst_error, 0.05)

    def test_super_premultiplied_fold_overflow_disables_candidate(self):
        surfaces = [
            Surface(1.0, (0.9, 0.9, 0.9, 0.1), id=1),  # rgb >> a, 超预乘
            Surface(1.0, (0.9, 0.9, 0.9, 0.1), id=2),
        ]
        res = simplify_surfaces(surfaces, tolerance=1.0)
        self.assertEqual(res.output_count, 2)
        self.assertEqual(res.surfaces, tuple(surfaces))
        c, _a = fold(surfaces)
        self.assertGreater(c[0], 1.0)

    def test_result_feeds_back_into_composite(self):
        surfaces = [
            Surface(1.0, (0.10, 0.20, 0.05, 0.20), id=1),
            Surface(1.5, (0.05, 0.05, 0.20, 0.10), id=2),
            Surface(2.0, (0.30, 0.10, 0.10, 0.50), id=3),
            Surface(2.0, (0.05, 0.10, 0.05, 0.10), id=4),
            Surface(3.0, (0.02, 0.02, 0.02, 0.05), id=5),
        ]
        res = simplify_surfaces(surfaces, tolerance=0.05)
        before = composite(surfaces)
        after = composite(list(res.surfaces))
        for b, a in zip(before.color, after.color):
            self.assertAlmostEqual(b, a, delta=0.05 + 1e-12)
        self.assertAlmostEqual(before.transmittance, after.transmittance, delta=1e-12)
        self.assertAlmostEqual(
            sum(e.contribution[0] for e in after.ledger), after.color[0], delta=1e-15
        )

    def test_input_not_modified_and_order_irrelevant(self):
        surfaces = [
            Surface(3.0, (0.1, 0.1, 0.1, 0.2), id=3),
            Surface(1.0, (0.2, 0.1, 0.1, 0.3), id=1),
            Surface(2.0, (0.1, 0.2, 0.1, 0.1), id=2),
        ]
        snapshot = list(surfaces)
        res = simplify_surfaces(surfaces, tolerance=0.01)
        self.assertEqual(surfaces, snapshot)
        res2 = simplify_surfaces(list(reversed(surfaces)), tolerance=0.01)
        self.assertEqual(res.surfaces, res2.surfaces)

    def test_empty_input(self):
        res = simplify_surfaces([], tolerance=0.1)
        self.assertEqual(res.surfaces, ())
        self.assertIsNone(res.worst_depth)
        self.assertEqual(res.worst_error, 0.0)
        self.assertEqual(res.output_count, 0)

    def test_rejects_volume_segment(self):
        with self.assertRaises(ValidationError):
            simplify_surfaces(
                [VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), 0.5)], tolerance=0.1
            )

    def test_rejects_bad_tolerance(self):
        s = [Surface(1.0, (0.1, 0.1, 0.1, 0.1))]
        for bad in (-0.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    simplify_surfaces(s, tolerance=bad)

    def test_item_limit(self):
        surfaces = [Surface(float(i), (0.01, 0.01, 0.01, 0.01)) for i in range(MAX_ITEMS + 1)]
        with self.assertRaises(ItemLimitError):
            simplify_surfaces(surfaces, tolerance=0.1)

    def test_worst_position_reported(self):
        surfaces = [
            Surface(1.0, (0.10, 0.0, 0.0, 0.2), id=1),
            Surface(2.0, (0.0, 0.3, 0.0, 0.4), id=2),
            Surface(3.0, (0.0, 0.0, 0.2, 0.2), id=3),
        ]
        res = simplify_surfaces(surfaces, tolerance=0.5)
        # 全并后透过率误差(边界 z=2): A_partial=0.52 > 0.5, 故最优为 2 个面。
        self.assertEqual(res.output_count, 2)
        self.assertIn(res.worst_depth, (1.0, 2.0, 3.0))
        zs = [1.0, 2.0, 3.0]
        before = states_at(surfaces, zs)
        after = states_at(list(res.surfaces), zs)
        want = max(
            max(abs(before[z][i] - after[z][i]) for i in range(4))
            for z in zs
        )
        self.assertAlmostEqual(res.worst_error, want, delta=1e-15)


if __name__ == "__main__":
    unittest.main()

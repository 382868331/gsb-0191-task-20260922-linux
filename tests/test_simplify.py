"""simplify_surfaces 单元测试。

最优性用独立的全枚举暴力解核对(不调用被测核心得到期望值);
往返一致性(压缩结果送回 composite)仅作补充检查。
"""

import itertools
import unittest

from deep_composite import (
    MAX_ITEMS,
    ItemLimitError,
    Surface,
    ValidationError,
    VolumeSegment,
    composite,
    simplify_surfaces,
)

EPS = 1e-9


def fold(seg):
    """独立手算折叠: C = sum T_i c_i, A = 1 - prod (1-a_i)。"""
    acc = [0.0, 0.0, 0.0]
    t = 1.0
    for s in seg:
        r, g, b, a = s.rgba
        for k in range(3):
            acc[k] += t * (r, g, b)[k]
        t *= 1.0 - a
    return tuple(acc), 1.0 - t


def prefix_at(surfaces, z):
    """独立手算: 深度边界 z 之后的前缀颜色与透过率。"""
    acc = [0.0, 0.0, 0.0]
    t = 1.0
    for s in sorted(surfaces, key=lambda s: (s.z, (0, s.id) if isinstance(s.id, int) else (1, s.id))):
        if s.z > z:
            break
        r, g, b, a = s.rgba
        for k in range(3):
            acc[k] += t * (r, g, b)[k]
        t *= 1.0 - a
    return tuple(acc), t


def brute_force_best(items, tol):
    """全枚举合法分组, 返回 (最少样本数, 字典序最小终点序列)。

    与被测实现独立的参考: 直接按题面定义枚举所有深度分界方案。
    """
    depths = sorted({s.z for s in items})
    groups = [[s for s in items if s.z == z] for z in depths]
    m = len(groups)
    best = None
    # 每个相邻深度间隙可选"分界/不分界"; 每组再选 折叠/原样保留(仅单深度组可保留)。
    for cuts in itertools.product([False, True], repeat=m - 1):
        segs = []
        start = 0
        for k in range(m - 1):
            if cuts[k]:
                segs.append((start, k + 1))
                start = k + 1
        segs.append((start, m))
        choices = []
        for a, b in segs:
            opts = ["fold"]
            if b - a == 1:
                opts.append("keep")
            choices.append(opts)
        for modes in itertools.product(*choices):
            out = []
            ok = True
            ends = []
            for (a, b), mode in zip(segs, modes):
                seg = [s for k in range(a, b) for s in groups[k]]
                if mode == "keep":
                    out.extend(seg)
                    ends.append(depths[b - 1])
                    continue
                rgb, alpha = fold(seg)
                if any(c > 1.0 + 1e-12 for c in rgb):
                    ok = False
                    break
                out.append(Surface(depths[b - 1], (rgb[0], rgb[1], rgb[2], alpha), id=seg[-1].id))
                ends.append(depths[b - 1])
            if not ok:
                continue
            # 误差核对: 每个原深度边界
            for z in depths:
                (oc, ot), (nc, nt) = prefix_at(items, z), prefix_at(out, z)
                err = max(
                    abs(oc[0] - nc[0]), abs(oc[1] - nc[1]),
                    abs(oc[2] - nc[2]), abs(ot - nt),
                )
                if err > tol + 1e-12:
                    ok = False
                    break
            if not ok:
                continue
            key = (len(out), tuple(ends))
            if best is None or key < best:
                best = key
    return best


def make_items(spec):
    """spec: [(z, rgba, id), ...]"""
    return [Surface(z, rgba, id=i) for i, (z, rgba, _) in enumerate(spec)]


class TestBruteForceOptimality(unittest.TestCase):
    """小序列全枚举核对: 样本数与平局终点序列都必须与暴力参考一致。"""

    def check_against_brute(self, items, tol):
        res = simplify_surfaces(items, tolerance=tol)
        want = brute_force_best(items, tol)
        self.assertIsNotNone(want)
        self.assertEqual(len(res.surfaces), want[0], f"tol={tol}")
        self.assertEqual(res.group_ends, want[1], f"tol={tol}")
        self.assertLessEqual(res.max_error, tol + 1e-9)

    def test_random_like_small_sequences(self):
        specs = [
            [(0.0, (0.02, 0.01, 0.03, 0.10), 0),
             (1.0, (0.03, 0.02, 0.01, 0.20), 0),
             (2.0, (0.01, 0.04, 0.02, 0.05), 0),
             (3.0, (0.02, 0.02, 0.02, 0.30), 0)],
            [(0.0, (0.10, 0.00, 0.00, 0.50), 0),
             (0.5, (0.00, 0.10, 0.00, 0.05), 0),
             (1.5, (0.00, 0.00, 0.10, 0.05), 0),
             (2.5, (0.05, 0.05, 0.00, 0.40), 0),
             (4.0, (0.01, 0.01, 0.01, 0.01), 0)],
        ]
        for spec in specs:
            items = make_items(spec)
            for tol in (0.0, 0.01, 0.05, 0.2, 0.5):
                self.check_against_brute(items, tol)

    def test_same_depth_never_split(self):
        """同深度 Surface 不拆开; 同深度折叠为零误差压缩。"""
        items = [
            Surface(1.0, (0.2, 0.1, 0.0, 0.5), id=2),
            Surface(1.0, (0.1, 0.2, 0.1, 0.5), id=1),
            Surface(1.0, (0.0, 0.1, 0.2, 0.5), id=3),
        ]
        res = simplify_surfaces(items, tolerance=0.0)
        self.assertEqual(len(res.surfaces), 1)  # 三个同深度折叠成一个
        self.assertEqual(res.max_error, 0.0)
        rgb, alpha = fold(sorted(items, key=lambda s: s.id))
        self.assertEqual(res.surfaces[0].z, 1.0)
        for got, want in zip(res.surfaces[0].rgba, rgb + (alpha,)):
            self.assertAlmostEqual(got, want, delta=EPS)

    def test_tolerance_zero_blocks_lossy_merge(self):
        """阈值 0: 跨深度折叠只要有非零误差就必须禁用。"""
        items = [
            Surface(1.0, (0.3, 0.0, 0.0, 0.5), id=1),
            Surface(2.0, (0.0, 0.3, 0.0, 0.5), id=2),
        ]
        res = simplify_surfaces(items, tolerance=0.0)
        self.assertEqual(len(res.surfaces), 2)
        self.assertEqual(res.max_error, 0.0)

    def test_transparent_sandwich_around_nearly_opaque(self):
        """高透明层夹着近乎不透明层: 误差主要来自被推迟的透明层。"""
        items = [
            Surface(1.0, (0.04, 0.04, 0.04, 0.05), id=1),   # 高透明
            Surface(2.0, (0.90, 0.90, 0.90, 0.999), id=2),  # 近乎不透明
            Surface(3.0, (0.04, 0.04, 0.04, 0.05), id=3),   # 高透明
        ]
        # 预算容得下把前面的高透明层并入近不透明层(仅推迟该透明层)
        res = simplify_surfaces(items, tolerance=0.05)
        self.assertEqual(len(res.surfaces), 2)
        self.assertEqual(res.group_ends, (2.0, 3.0))
        self.assertLessEqual(res.max_error, 0.05 + 1e-9)
        # 大预算可全折叠; 但中间边界会出现近不透明层被推迟的大误差, 小预算不行
        res_big = simplify_surfaces(items, tolerance=1.0)
        self.assertEqual(len(res_big.surfaces), 1)
        res = simplify_surfaces(items, tolerance=0.001)
        self.assertEqual(len(res.surfaces), 3)
        self.assertEqual(res.max_error, 0.0)

    def test_super_premultiplied_overflow_disables_fold(self):
        """超预乘输入折叠颜色越界: 候选段禁用, 不钳制颜色。"""
        items = [
            Surface(1.0, (1.0, 0.0, 0.0, 0.0), id=1),
            Surface(2.0, (1.0, 0.0, 0.0, 0.0), id=2),
        ]
        res = simplify_surfaces(items, tolerance=100.0)
        self.assertEqual(len(res.surfaces), 2)  # 折叠色 r=2 越界, 禁用
        for s in res.surfaces:
            self.assertEqual(s.rgba[0], 1.0)  # 原样保留, 未被钳制

    def test_roundtrip_through_composite(self):
        """压缩结果可直接送回 composite, 且与原始合成在预算内一致。"""
        items = [Surface(float(i), (0.02, 0.03, 0.01, 0.05), id=i) for i in range(8)]
        tol = 0.05
        res = simplify_surfaces(items, tolerance=tol)
        before = composite(items)
        after = composite(res.surfaces)
        for k in range(3):
            self.assertAlmostEqual(before.color[k], after.color[k], delta=tol + 1e-9)
        self.assertAlmostEqual(before.transmittance, after.transmittance, delta=tol + 1e-9)
        # 原输入未被修改
        for i, s in enumerate(items):
            self.assertEqual(s.z, float(i))
            self.assertEqual(s.id, i)

    def test_worst_location_reported(self):
        items = [Surface(float(i), (0.02, 0.02, 0.02, 0.02), id=i) for i in range(6)]
        res = simplify_surfaces(items, tolerance=0.03)
        self.assertIsNotNone(res.worst_z)
        # 报告的最坏位置处误差确实达到 max_error
        oc, ot = prefix_at(items, res.worst_z)
        nc, nt = prefix_at(list(res.surfaces), res.worst_z)
        err = max(abs(oc[0] - nc[0]), abs(oc[1] - nc[1]),
                  abs(oc[2] - nc[2]), abs(ot - nt))
        self.assertAlmostEqual(err, res.max_error, delta=1e-12)


class TestSimplifyValidation(unittest.TestCase):
    def test_rejects_volume_segment(self):
        with self.assertRaises(ValidationError):
            simplify_surfaces([VolumeSegment(0.0, 1.0, (0.1, 0.1, 0.1), 0.5)], tolerance=0.1)

    def test_rejects_bad_tolerance(self):
        for bad in (float("nan"), float("inf"), -0.1, True):
            with self.assertRaises(ValidationError, msg=f"tol={bad}"):
                simplify_surfaces([Surface(1.0, (0.1, 0.1, 0.1, 0.1))], tolerance=bad)

    def test_item_limit(self):
        items = [Surface(float(i), (0.01, 0.01, 0.01, 0.01), id=i) for i in range(MAX_ITEMS + 1)]
        with self.assertRaises(ItemLimitError):
            simplify_surfaces(items, tolerance=0.1)

    def test_empty_input(self):
        res = simplify_surfaces([], tolerance=0.1)
        self.assertEqual(res.surfaces, ())
        self.assertEqual(res.max_error, 0.0)
        self.assertIsNone(res.worst_z)
        self.assertEqual(res.group_ends, ())

    def test_input_not_mutated_and_order_independent(self):
        items = [
            Surface(3.0, (0.02, 0.02, 0.02, 0.02), id=3),
            Surface(1.0, (0.02, 0.02, 0.02, 0.02), id=1),
            Surface(2.0, (0.02, 0.02, 0.02, 0.02), id=2),
        ]
        snapshot = [(s.z, s.rgba, s.id) for s in items]
        res = simplify_surfaces(iter(items), tolerance=0.05)  # 迭代器输入也可
        self.assertEqual([(s.z, s.rgba, s.id) for s in items], snapshot)
        self.assertEqual([s.z for s in res.surfaces], sorted(s.z for s in res.surfaces))


if __name__ == "__main__":
    unittest.main()

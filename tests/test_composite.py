"""Tests for deepcomposite.

Analytic reference values are computed directly with ``math.exp`` /
``math.expm1`` inside this file; the expected values never come from
calling ``deepcomposite.composite``.  Round-trip / invariance checks
(e.g. segmentation invariance) are only used as supplementary
consistency checks alongside the analytic references.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepcomposite import (
    MAX_ITEMS,
    DeepCompositeError,
    Surface,
    VolumeSegment,
    composite,
)

TOL = 1e-12  # absolute tolerance vs analytic references (O(1) magnitudes)


def slab_color(e, sigma, length):
    """Independent reference: emitted radiance of one uniform slab."""
    if sigma == 0.0:
        return e * length
    return e / sigma * (1.0 - math.exp(-sigma * length))


def slab_trans(sigma, length):
    return math.exp(-sigma * length)


def assert_vec3_close(tc, got, want, tol=TOL):
    tc.assertEqual(len(got), 3)
    for g, w in zip(got, want):
        tc.assertAlmostEqual(g, w, delta=tol)


class TestAnalyticReferences(unittest.TestCase):
    def test_single_segment(self):
        # One uniform segment, mixed zero / non-zero sigma per channel.
        e = (0.5, 0.25, 0.0)
        sig = (0.5, 0.0, 1.0)
        res = composite([VolumeSegment(1, 0.0, 2.0, e, sig)])
        want_color = tuple(slab_color(e[c], sig[c], 2.0) for c in range(3))
        want_trans = tuple(slab_trans(sig[c], 2.0) for c in range(3))
        assert_vec3_close(self, res.color, want_color)
        assert_vec3_close(self, res.transmittance, want_trans)
        # Sanity: exact literals of the reference values.
        self.assertAlmostEqual(res.color[0], 1.0 - math.exp(-1.0), delta=TOL)
        self.assertAlmostEqual(res.color[1], 0.5, delta=TOL)
        self.assertAlmostEqual(res.color[2], 0.0, delta=TOL)
        self.assertEqual(len(res.ledger), 1)
        self.assertEqual(res.ledger[0].kind, "segment")
        self.assertEqual((res.ledger[0].z0, res.ledger[0].z1), (0.0, 2.0))

    def test_double_overlap_slabs(self):
        # A over [0,2], B over [1,3] -> slabs [0,1], [1,2], [2,3].
        e_a = (0.3, 0.1, 0.2)
        s_a = (0.4, 0.0, 0.5)
        e_b = (0.2, 0.6, 0.1)
        s_b = (0.1, 0.3, 0.0)
        res = composite(
            [
                VolumeSegment(1, 0.0, 2.0, e_a, s_a),
                VolumeSegment(2, 1.0, 3.0, e_b, s_b),
            ]
        )
        slabs = [
            (e_a, s_a),
            (tuple(e_a[c] + e_b[c] for c in range(3)),
             tuple(s_a[c] + s_b[c] for c in range(3))),
            (e_b, s_b),
        ]
        want_color = [0.0, 0.0, 0.0]
        want_trans = [1.0, 1.0, 1.0]
        for e, s in slabs:
            for c in range(3):
                want_color[c] += want_trans[c] * slab_color(e[c], s[c], 1.0)
                want_trans[c] *= slab_trans(s[c], 1.0)
        assert_vec3_close(self, res.color, want_color)
        assert_vec3_close(self, res.transmittance, want_trans)
        # Ledger must show exactly the three non-overlapping slabs, with
        # both segments contributing in the middle slab (sorted by id).
        spans = [(en.item_id, en.z0, en.z1) for en in res.ledger]
        self.assertEqual(
            spans,
            [(1, 0.0, 1.0), (1, 1.0, 2.0), (2, 1.0, 2.0), (2, 2.0, 3.0)],
        )

    def test_segmentation_invariance(self):
        # Supplementary consistency check: one segment vs two adjacent
        # segments with identical coefficients must agree.
        e = (0.4, 0.2, 0.6)
        sig = (0.3, 0.0, 0.9)
        whole = composite([VolumeSegment(1, 0.0, 2.0, e, sig)])
        split = composite(
            [
                VolumeSegment(1, 0.0, 1.0, e, sig),
                VolumeSegment(2, 1.0, 2.0, e, sig),
            ]
        )
        assert_vec3_close(self, split.color, whole.color)
        assert_vec3_close(self, split.transmittance, whole.transmittance)
        # And the whole-segment result still matches the analytic reference.
        want = tuple(slab_color(e[c], sig[c], 2.0) for c in range(3))
        assert_vec3_close(self, whole.color, want)


class TestBoundaries(unittest.TestCase):
    def test_zero_extinction(self):
        # sigma == 0 everywhere: C = e * length, transmittance stays 1.
        e = (0.3, 0.5, 0.7)
        res = composite([VolumeSegment(7, 1.0, 3.5, e, (0.0, 0.0, 0.0))])
        assert_vec3_close(self, res.color, tuple(2.5 * v for v in e))
        assert_vec3_close(self, res.transmittance, (1.0, 1.0, 1.0))

    def test_same_depth_surfaces_sorted_by_id(self):
        # Same z: smaller id composites first (nearer), regardless of the
        # order the items are passed in.
        s1 = Surface(1, 1.0, (0.2, 0.1, 0.05, 0.5))
        s2 = Surface(2, 1.0, (0.12, 0.24, 0.36, 0.6))
        res = composite([s2, s1])
        want_color = tuple(
            s1.rgba[c] + (1.0 - s1.rgba[3]) * s2.rgba[c] for c in range(3)
        )
        want_trans = ((1.0 - 0.5) * (1.0 - 0.6),) * 3
        assert_vec3_close(self, res.color, want_color)
        assert_vec3_close(self, res.transmittance, want_trans)
        self.assertEqual([en.item_id for en in res.ledger], [1, 2])

    def test_surface_between_segments_at_same_z(self):
        # Segments ending at z finish before the surface; segments starting
        # at z begin after it.  Order matters because the surface absorbs.
        seg_a = VolumeSegment(1, 0.0, 1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        surf = Surface(5, 1.0, (0.0, 0.5, 0.0, 0.5))
        seg_b = VolumeSegment(2, 1.0, 2.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        res = composite([seg_b, surf, seg_a])
        # After A: color (1,0,0), T=(1,1,1).
        # Surface: color += (0,0.5,0); T=(0.5,0.5,0.5) (alpha absorbs all
        # channels equally).
        # After B: color += T*(0,0,1) = (0,0,0.5).
        assert_vec3_close(self, res.color, (1.0, 0.5, 0.5))
        assert_vec3_close(self, res.transmittance, (0.5, 0.5, 0.5))
        self.assertEqual(
            [en.kind for en in res.ledger],
            ["segment", "surface", "segment"],
        )

    def test_truncated_far_plane(self):
        e = (0.1, 0.2, 0.3)
        sig = (0.2, 0.1, 0.0)
        res = composite([VolumeSegment(1, 0.0, 10.0, e, sig)], z_far=4.0)
        want_color = tuple(slab_color(e[c], sig[c], 4.0) for c in range(3))
        want_trans = tuple(slab_trans(sig[c], 4.0) for c in range(3))
        assert_vec3_close(self, res.color, want_color)
        assert_vec3_close(self, res.transmittance, want_trans)
        self.assertEqual((res.ledger[0].z0, res.ledger[0].z1), (0.0, 4.0))

    def test_far_plane_drops_and_keeps_surfaces(self):
        near = Surface(1, 4.0, (0.1, 0.1, 0.1, 0.5))  # z == z_far: kept
        far = Surface(2, 5.0, (0.9, 0.9, 0.9, 0.9))   # z > z_far: dropped
        res = composite([near, far], z_far=4.0)
        assert_vec3_close(self, res.color, (0.1, 0.1, 0.1))
        assert_vec3_close(self, res.transmittance, (0.5, 0.5, 0.5))
        self.assertEqual([en.item_id for en in res.ledger], [1])

    def test_empty_pixel(self):
        res = composite([])
        self.assertEqual(res.color, (0.0, 0.0, 0.0))
        self.assertEqual(res.transmittance, (1.0, 1.0, 1.0))
        self.assertEqual(res.ledger, ())

    def test_contribution_additivity(self):
        # Supplementary consistency check: ledger contributions must sum
        # to the final color on a mixed scene.
        scene = [
            VolumeSegment(10, 0.0, 3.0, (0.05, 0.04, 0.03), (0.2, 0.25, 0.3)),
            VolumeSegment(11, 1.0, 2.0, (0.8, 0.3, 0.1), (0.1, 0.1, 0.1)),
            Surface(1, 0.5, (0.3, 0.1, 0.1, 0.5)),
            Surface(2, 2.5, (0.05, 0.2, 0.4, 0.8)),
        ]
        res = composite(scene)
        summed = [0.0, 0.0, 0.0]
        for en in res.ledger:
            for c in range(3):
                summed[c] += en.contribution[c]
        assert_vec3_close(self, tuple(summed), res.color)
        self.assertEqual({en.item_id for en in res.ledger}, {1, 2, 10, 11})


class TestRejections(unittest.TestCase):
    def assert_rejected(self, items, needle, **kwargs):
        with self.assertRaises(DeepCompositeError) as ctx:
            composite(items, **kwargs)
        self.assertIn(needle, str(ctx.exception))

    def test_negative_sigma(self):
        self.assert_rejected(
            [VolumeSegment(1, 0.0, 1.0, (0, 0, 0), (-0.1, 0, 0))],
            "items[0].sigma[0]: negative sigma rejected",
        )

    def test_negative_emission(self):
        self.assert_rejected(
            [VolumeSegment(1, 0.0, 1.0, (0, -1.0, 0), (0, 0, 0))],
            "items[0].emission[1]: negative emission rejected",
        )

    def test_nan_and_inf(self):
        self.assert_rejected(
            [Surface(1, float("nan"), (0.0, 0.0, 0.0, 0.0))], "items[0].z"
        )
        self.assert_rejected(
            [VolumeSegment(1, 0.0, 1.0, (float("inf"), 0, 0), (0, 0, 0))],
            "items[0].emission[0]",
        )
        self.assert_rejected(
            [VolumeSegment(1, 0.0, 1.0, (0, 0, 0), (0, 0, float("-inf")))],
            "items[0].sigma[2]",
        )

    def test_bool_rejected_for_int_and_float_params(self):
        self.assert_rejected(
            [Surface(True, 0.5, (0.0, 0.0, 0.0, 0.0))], "items[0].id"
        )
        self.assert_rejected(
            [Surface(1, True, (0.0, 0.0, 0.0, 0.0))], "items[0].z"
        )
        self.assert_rejected([], "z_far", z_far=True)

    def test_premultiplied_convention_enforced(self):
        # channel > alpha violates the premultiplied contract
        self.assert_rejected(
            [Surface(1, 0.5, (0.5, 0.0, 0.0, 0.4))], "items[0].rgba[0]"
        )
        # alpha out of range
        self.assert_rejected(
            [Surface(1, 0.5, (0.0, 0.0, 0.0, 1.5))], "items[0].rgba[3]"
        )
        self.assert_rejected(
            [Surface(1, 0.5, (-0.1, 0.0, 0.0, 0.4))], "items[0].rgba[0]"
        )

    def test_degenerate_segment(self):
        self.assert_rejected(
            [VolumeSegment(1, 1.0, 1.0, (0, 0, 0), (0, 0, 0))], "z1 > z0"
        )
        self.assert_rejected(
            [VolumeSegment(1, 2.0, 1.0, (0, 0, 0), (0, 0, 0))], "z1 > z0"
        )

    def test_item_limit(self):
        items = [Surface(i, float(i), (0.0, 0.0, 0.0, 0.0))
                 for i in range(MAX_ITEMS + 1)]
        self.assert_rejected(items, "MAX_ITEMS")
        # Exactly MAX_ITEMS is accepted.
        res = composite(items[:MAX_ITEMS])
        self.assertEqual(len(res.ledger), MAX_ITEMS)

    def test_wrong_item_type(self):
        self.assert_rejected([object()], "items[0]")
        self.assert_rejected([("surface", 1, 0.5)], "items[0]")

    def test_bad_vector_shapes(self):
        self.assert_rejected(
            [VolumeSegment(1, 0.0, 1.0, (0.0, 0.0), (0, 0, 0))],
            "items[0].emission",
        )
        self.assert_rejected(
            [Surface(1, 0.5, (0.0, 0.0, 0.0))], "items[0].rgba"
        )

    def test_non_finite_z_far(self):
        self.assert_rejected([], "z_far", z_far=float("nan"))

    def test_error_is_value_error(self):
        with self.assertRaises(ValueError):
            composite([VolumeSegment(1, 0.0, 1.0, (0, 0, 0), (-1, 0, 0))])


if __name__ == "__main__":
    unittest.main()

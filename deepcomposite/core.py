"""Two-layer deep-sample transparent compositing (per-pixel, front-to-back).

Conventions
-----------
* Camera looks toward increasing ``z``; smaller ``z`` is nearer.
* All colors are **premultiplied**: a :class:`Surface` carries ``rgba`` with
  ``0 <= r, g, b <= a <= 1`` (channel values already multiplied by alpha).
* Volume segments are uniform over ``[z0, z1)`` with per-channel emission
  coefficient ``emission`` and extinction coefficient ``sigma``.
* Transmittance is per-channel (RGB), because ``sigma`` is per-channel.

Compositing rules (fixed public contract)
-----------------------------------------
* Segment endpoints split the depth axis into non-overlapping slabs; inside a
  slab, ``sigma`` of overlapping segments sums and ``emission`` sums.
* Slab emission integral per channel:
  ``C = e / sigma * (1 - exp(-sigma * length))``, with ``C = e * length``
  when ``sigma == 0``.  Slab transmittance is ``exp(-sigma * length)``.
* Surfaces sort by ``(z, id)``.  At a shared depth, segments ending there
  finish first, then the surface composites, then segments starting there
  begin.
* Negative ``sigma`` (and negative ``emission``) is rejected, never clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

__all__ = [
    "MAX_ITEMS",
    "DeepCompositeError",
    "Surface",
    "VolumeSegment",
    "LedgerEntry",
    "CompositeResult",
    "composite",
]

#: Maximum number of deep samples (surfaces + volume segments) per pixel.
MAX_ITEMS = 100

Vec3 = Tuple[float, float, float]


class DeepCompositeError(ValueError):
    """Raised for invalid input to :func:`composite`.

    The message always locates the offending item and field, e.g.
    ``items[3].sigma[1]: negative sigma rejected (got -0.5)``.
    It subclasses :class:`ValueError` so generic value-error handling
    also catches it.
    """


@dataclass(frozen=True)
class Surface:
    """Infinitesimally thin opaque-ish layer at depth ``z``.

    ``rgba`` is premultiplied: ``0 <= r, g, b <= a <= 1``.
    """

    id: int
    z: float
    rgba: Tuple[float, float, float, float]


@dataclass(frozen=True)
class VolumeSegment:
    """Uniform volume over ``[z0, z1)`` (requires ``z1 > z0``).

    ``emission`` and ``sigma`` are per-channel non-negative coefficients.
    """

    id: int
    z0: float
    z1: float
    emission: Vec3
    sigma: Vec3


@dataclass(frozen=True)
class LedgerEntry:
    """One compositing step's contribution to the final color.

    ``contribution`` is already weighted by the transmittance in front of
    the item, so ``sum(entry.contribution) == CompositeResult.color``
    (up to floating-point rounding).  For a volume segment, ``z0``/``z1``
    are the bounds of the slab this entry covers; one segment may produce
    several entries when it overlaps others.  For a surface, ``z0 == z1``.
    """

    kind: str  # "surface" or "segment"
    item_id: int
    z0: float
    z1: float
    contribution: Vec3
    transmittance_before: Vec3


@dataclass(frozen=True)
class CompositeResult:
    """Result of :func:`composite`.

    ``color`` is the accumulated premultiplied RGB; ``transmittance`` is
    the per-channel fraction of background light passing through the whole
    pixel; ``ledger`` lists every contribution in compositing order.
    """

    color: Vec3
    transmittance: Vec3
    ledger: Tuple[LedgerEntry, ...]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _finite(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeepCompositeError(
            f"{where}: expected a real number (int/float, bool rejected), "
            f"got {type(value).__name__}"
        )
    v = float(value)
    if not math.isfinite(v):
        raise DeepCompositeError(
            f"{where}: NaN/Infinity rejected (got {value!r})"
        )
    return v


def _item_id(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeepCompositeError(
            f"{where}: id must be an int (bool rejected), got {value!r}"
        )
    return value


def _vec(values, where: str, n: int) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise DeepCompositeError(f"{where}: expected a sequence of {n} numbers")
    try:
        vals = tuple(values)
    except TypeError:
        raise DeepCompositeError(
            f"{where}: expected a sequence of {n} numbers, "
            f"got {type(values).__name__}"
        ) from None
    if len(vals) != n:
        raise DeepCompositeError(
            f"{where}: expected {n} components, got {len(vals)}"
        )
    return tuple(_finite(v, f"{where}[{i}]") for i, v in enumerate(vals))


def _validate(item, index: int):
    """Validate one item; returns a normalized copy.  Pure: no side effects."""
    where = f"items[{index}]"
    if isinstance(item, Surface):
        iid = _item_id(item.id, f"{where}.id")
        z = _finite(item.z, f"{where}.z")
        r, g, b, a = _vec(item.rgba, f"{where}.rgba", 4)
        if not 0.0 <= a <= 1.0:
            raise DeepCompositeError(
                f"{where}.rgba[3]: alpha must be in [0, 1], got {a!r}"
            )
        for i, c in enumerate((r, g, b)):
            if not 0.0 <= c <= a:
                raise DeepCompositeError(
                    f"{where}.rgba[{i}]: premultiplied channel must satisfy "
                    f"0 <= c <= alpha ({c!r} vs alpha {a!r})"
                )
        return Surface(iid, z, (r, g, b, a))
    if isinstance(item, VolumeSegment):
        iid = _item_id(item.id, f"{where}.id")
        z0 = _finite(item.z0, f"{where}.z0")
        z1 = _finite(item.z1, f"{where}.z1")
        emission = _vec(item.emission, f"{where}.emission", 3)
        sigma = _vec(item.sigma, f"{where}.sigma", 3)
        for i, e in enumerate(emission):
            if e < 0.0:
                raise DeepCompositeError(
                    f"{where}.emission[{i}]: negative emission rejected "
                    f"(got {e!r})"
                )
        for i, s in enumerate(sigma):
            if s < 0.0:
                raise DeepCompositeError(
                    f"{where}.sigma[{i}]: negative sigma rejected (got {s!r})"
                )
        if not z1 > z0:
            raise DeepCompositeError(
                f"{where}: require z1 > z0 (got z0={z0!r}, z1={z1!r})"
            )
        return VolumeSegment(iid, z0, z1, emission, sigma)
    raise DeepCompositeError(
        f"{where}: expected Surface or VolumeSegment, "
        f"got {type(item).__name__}"
    )


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def _slab_piece(e: float, sigma_total: float, length: float) -> float:
    """One segment's share of a slab's emitted radiance (per channel)."""
    if e == 0.0:
        return 0.0
    if sigma_total == 0.0:
        return e * length
    # -expm1(-x) == 1 - exp(-x), accurate for small x.
    return e / sigma_total * (-math.expm1(-sigma_total * length))


def composite(
    items: Iterable,
    *,
    z_far: Optional[float] = None,
) -> CompositeResult:
    """Composite one pixel's deep samples front-to-back.

    ``items`` is an iterable of at most :data:`MAX_ITEMS` :class:`Surface`
    / :class:`VolumeSegment` objects.  ``z_far`` optionally truncates the
    pixel: surfaces with ``z > z_far`` are dropped and segments are clipped
    to ``z1 <= z_far`` (segments entirely beyond are dropped).

    The whole batch is validated before any compositing happens, so a
    rejected call produces no partial result.  The function is pure.
    """
    items = list(items)
    if len(items) > MAX_ITEMS:
        raise DeepCompositeError(
            f"items: {len(items)} items exceed MAX_ITEMS ({MAX_ITEMS})"
        )
    if z_far is not None:
        z_far = _finite(z_far, "z_far")

    # Validate everything up front: a failing batch leaves no partial state.
    valid = [_validate(it, i) for i, it in enumerate(items)]

    surfaces = []
    segments = []
    for it in valid:
        if isinstance(it, Surface):
            if z_far is None or it.z <= z_far:
                surfaces.append(it)
        else:
            z1 = it.z1 if z_far is None else min(it.z1, z_far)
            if z1 > it.z0:
                segments.append(
                    VolumeSegment(it.id, it.z0, z1, it.emission, it.sigma)
                )

    surfaces.sort(key=lambda s: (s.z, s.id))
    breakpoints = sorted(
        {s.z for s in surfaces}
        | {v.z0 for v in segments}
        | {v.z1 for v in segments}
    )

    color = [0.0, 0.0, 0.0]
    trans = [1.0, 1.0, 1.0]
    ledger = []

    def do_slab(a: float, b: float) -> None:
        length = b - a
        active = sorted(
            (s for s in segments if s.z0 <= a and s.z1 >= b),
            key=lambda s: s.id,
        )
        if not active:
            return
        sig = [sum(s.sigma[c] for s in active) for c in range(3)]
        t_before = (trans[0], trans[1], trans[2])
        for s in active:
            contrib = [0.0, 0.0, 0.0]
            for c in range(3):
                piece = _slab_piece(s.emission[c], sig[c], length)
                contrib[c] = trans[c] * piece
                color[c] += contrib[c]
            ledger.append(
                LedgerEntry("segment", s.id, a, b, tuple(contrib), t_before)
            )
        for c in range(3):
            trans[c] *= math.exp(-sig[c] * length)

    def do_surface(surf: Surface) -> None:
        t_before = (trans[0], trans[1], trans[2])
        r, g, b, a = surf.rgba
        contrib = (trans[0] * r, trans[1] * g, trans[2] * b)
        color[0] += contrib[0]
        color[1] += contrib[1]
        color[2] += contrib[2]
        keep = 1.0 - a
        for c in range(3):
            trans[c] *= keep
        ledger.append(
            LedgerEntry("surface", surf.id, surf.z, surf.z, contrib, t_before)
        )

    si = 0
    prev = None
    for bp in breakpoints:
        # Segments ending at bp finish before surfaces at bp; segments
        # starting at bp begin after them.
        if prev is not None and bp > prev:
            do_slab(prev, bp)
        while si < len(surfaces) and surfaces[si].z == bp:
            do_surface(surfaces[si])
            si += 1
        prev = bp

    return CompositeResult(tuple(color), tuple(trans), tuple(ledger))

"""Deterministic demo for the deepcomposite module.

Builds one fixed pixel of deep samples (two overlapping volume segments,
two surfaces), composites it, verifies that the contribution ledger sums
to the final color, then triggers a real rejection (negative sigma).
Runs in well under 8 seconds; no sleeps, no recorded results.
"""

from deepcomposite import (
    DeepCompositeError,
    Surface,
    VolumeSegment,
    composite,
)


def fmt(v):
    return "(" + ", ".join(f"{x:.6f}" for x in v) + ")"


def main():
    scene = [
        VolumeSegment(10, 0.0, 3.0, (0.05, 0.04, 0.03), (0.2, 0.25, 0.3)),
        VolumeSegment(11, 1.0, 2.0, (0.8, 0.3, 0.1), (0.1, 0.1, 0.1)),
        Surface(1, 0.5, (0.30, 0.10, 0.10, 0.5)),
        Surface(2, 2.5, (0.05, 0.20, 0.40, 0.8)),
    ]

    print("=== normal case: 4 deep samples (2 segments, 2 surfaces) ===")
    result = composite(scene)
    print(f"final color        : {fmt(result.color)}")
    print(f"transmittance      : {fmt(result.transmittance)}")
    print("contribution ledger (compositing order):")
    for en in result.ledger:
        span = f"z=[{en.z0:.1f}, {en.z1:.1f}]"
        print(
            f"  {en.kind:7s} id={en.item_id:2d} {span:15s} "
            f"contrib={fmt(en.contribution)} T_before={fmt(en.transmittance_before)}"
        )

    # Contribution additivity check: ledger must sum to the final color.
    summed = [0.0, 0.0, 0.0]
    for en in result.ledger:
        for c in range(3):
            summed[c] += en.contribution[c]
    max_diff = max(abs(summed[c] - result.color[c]) for c in range(3))
    print(f"ledger sum         : {fmt(summed)}")
    print(f"additivity max|diff|: {max_diff:.3e} -> "
          + ("OK" if max_diff <= 1e-12 else "MISMATCH"))

    print()
    print("=== rejection case: negative sigma ===")
    bad = [VolumeSegment(99, 0.0, 1.0, (0.1, 0.1, 0.1), (0.5, -0.2, 0.5))]
    try:
        composite(bad)
    except DeepCompositeError as exc:
        print(f"DeepCompositeError raised as expected: {exc}")
    else:
        print("ERROR: negative sigma was not rejected")


if __name__ == "__main__":
    main()

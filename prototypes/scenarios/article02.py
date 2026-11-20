"""Standalone XNOR/popcount correctness check matching the archived proof routine."""

from __future__ import annotations

import random


def _xnor_popcount_dot(left: list[int], right: list[int]) -> int:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have the same nonzero length")
    if any(int(value) not in (-1, 1) for value in (*left, *right)):
        raise ValueError("binary vectors must contain only -1 and +1")
    width = len(left)
    bits_left = sum((value > 0) << bit for bit, value in enumerate(left))
    bits_right = sum((value > 0) << bit for bit, value in enumerate(right))
    matches = (~(bits_left ^ bits_right) & ((1 << width) - 1)).bit_count()
    return 2 * matches - width


def run(seed: int = 42) -> dict:
    rng = random.Random(seed)
    sizes = [1, 7, 8, 9, 63, 64, 65, 127, 129, 257]
    checked = 0
    for size in sizes:
        for _ in range(100):
            left = [rng.choice((-1, 1)) for _ in range(size)]
            right = [rng.choice((-1, 1)) for _ in range(size)]
            expected = sum(a * b for a, b in zip(left, right))
            if _xnor_popcount_dot(left, right) != expected:
                raise AssertionError(f"binary dot mismatch at vector size {size}")
            checked += 1
    return {
        "evidence_type": "computed",
        "seed": seed,
        "vectors_checked": checked,
        "sizes": sizes,
        "status": "PASS",
        "limitations": [
            "This checks exact integer arithmetic only.",
            "It does not measure runtime speed or trained-model quality.",
        ],
    }

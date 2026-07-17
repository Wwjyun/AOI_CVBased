from __future__ import annotations

import argparse
import json
from time import perf_counter

import cv2
import numpy as np


def _make_mask(width: int, height: int, blob_count: int, seed: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(blob_count):
        x = int(rng.integers(0, max(1, width - 60)))
        y = int(rng.integers(0, max(1, height - 60)))
        blob_width = int(rng.integers(8, 60))
        blob_height = int(rng.integers(8, 60))
        cv2.rectangle(
            mask,
            (x, y),
            (min(x + blob_width, width - 1), min(y + blob_height, height - 1)),
            255,
            thickness=cv2.FILLED,
        )
    return mask


def _measure(operation, mask: np.ndarray, rounds: int) -> dict[str, float]:
    for _ in range(3):
        operation(mask)
    timings = []
    for _ in range(rounds):
        started = perf_counter()
        operation(mask)
        timings.append((perf_counter() - started) * 1000.0)
    return {
        "median_ms": float(np.median(timings)),
        "mean_ms": float(np.mean(timings)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CPU contour extraction strategies.")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--blobs", type=int, default=350)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=118)
    args = parser.parse_args()
    if min(args.width, args.height, args.blobs, args.rounds) <= 0:
        parser.error("width, height, blobs, and rounds must be positive")

    mask = _make_mask(args.width, args.height, args.blobs, args.seed)
    operations = {
        "find_contours_list": lambda image: cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        ),
        "connected_components_with_stats": lambda image: cv2.connectedComponentsWithStats(
            image, connectivity=8
        ),
    }
    result = {
        "shape": [args.height, args.width],
        "blob_count": args.blobs,
        "rounds": args.rounds,
        "seed": args.seed,
        "timings": {
            name: _measure(operation, mask, args.rounds)
            for name, operation in operations.items()
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

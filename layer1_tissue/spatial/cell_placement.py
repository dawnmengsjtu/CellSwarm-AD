# -*- coding: utf-8 -*-
"""Cell spatial placement strategies."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CellPlacer:
    """Place cells in 2D space with different distribution strategies."""
    width: int = 100
    height: int = 100

    def uniform(self, n: int, seed: int = 42) -> List[Tuple[int, int]]:
        """Random uniform placement."""
        rng = np.random.default_rng(seed)
        xs = rng.integers(0, self.width, size=n)
        ys = rng.integers(0, self.height, size=n)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def clustered(self, n: int, centers: List[Tuple[int, int]], sigma: float = 10.0, seed: int = 42) -> List[Tuple[int, int]]:
        """Gaussian-clustered placement around given centers."""
        rng = np.random.default_rng(seed)
        positions = []
        per_center = n // len(centers)
        remainder = n % len(centers)
        for i, (cx, cy) in enumerate(centers):
            count = per_center + (1 if i < remainder else 0)
            xs = rng.normal(cx, sigma, size=count).astype(int)
            ys = rng.normal(cy, sigma, size=count).astype(int)
            xs = np.clip(xs, 0, self.width - 1)
            ys = np.clip(ys, 0, self.height - 1)
            positions.extend([(int(x), int(y)) for x, y in zip(xs, ys)])
        return positions

    def layered(self, counts: List[int], y_ranges: List[Tuple[int, int]], seed: int = 42) -> List[Tuple[int, int]]:
        """Layered placement: different densities in different y-ranges."""
        rng = np.random.default_rng(seed)
        positions = []
        for count, (y_min, y_max) in zip(counts, y_ranges):
            xs = rng.integers(0, self.width, size=count)
            ys = rng.integers(y_min, min(y_max, self.height), size=count)
            positions.extend([(int(x), int(y)) for x, y in zip(xs, ys)])
        return positions

# -*- coding: utf-8 -*-
"""Importance sampler for focused parameter exploration."""
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional


@dataclass
class ImportanceSampler:
    """Importance sampling with weight-based resampling."""
    n_samples: int = 100
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

    def sample_with_weights(self, param_ranges: Dict[str, tuple],
                            weight_fn: Callable) -> List[Dict]:
        """Sample with importance weights."""
        raw_samples = []
        for _ in range(self.n_samples):
            point = {}
            for name, (lo, hi) in param_ranges.items():
                point[name] = random.uniform(lo, hi)
            w = weight_fn(point)
            raw_samples.append({"params": point, "weight": w})

        total_w = sum(s["weight"] for s in raw_samples) or 1.0
        for s in raw_samples:
            s["weight"] /= total_w

        return raw_samples

    def resample(self, weighted_samples: List[Dict], n: int = 50) -> List[Dict]:
        """Resample based on weights."""
        weights = [s["weight"] for s in weighted_samples]
        chosen = random.choices(weighted_samples, weights=weights, k=n)
        return [s["params"] for s in chosen]

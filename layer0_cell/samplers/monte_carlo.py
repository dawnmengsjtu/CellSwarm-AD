# -*- coding: utf-8 -*-
"""Monte Carlo sampler for cell parameter exploration."""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional


@dataclass
class MonteCarloSampler:
    """Basic Monte Carlo sampler for parameter space exploration."""
    n_samples: int = 100
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

    def sample(self, param_ranges: Dict[str, tuple]) -> List[Dict[str, float]]:
        """Sample parameter combinations uniformly."""
        samples = []
        for _ in range(self.n_samples):
            point = {}
            for name, (lo, hi) in param_ranges.items():
                point[name] = random.uniform(lo, hi)
            samples.append(point)
        return samples

    def evaluate(self, samples: List[Dict], score_fn: Callable) -> List[Dict]:
        """Evaluate samples with a scoring function."""
        results = []
        for s in samples:
            score = score_fn(s)
            results.append({"params": s, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

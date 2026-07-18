# -*- coding: utf-8 -*-
"""Parameter optimizer for fitting pathology models to data."""
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple
from collections import deque
import random
import math


@dataclass
class ObjectiveFunction:
    """Wraps a callable objective for optimization."""
    name: str
    func: Callable[[Dict[str, float]], float]
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        # Support List[Tuple] for backward compatibility
        if isinstance(self.param_bounds, list):
            self.param_bounds = {str(i): bounds for i, bounds in enumerate(self.param_bounds)}

    def evaluate(self, params: Dict[str, float]) -> float:
        # Support both Dict and List for func
        if isinstance(params, dict) and all(isinstance(k, str) and k.isdigit() for k in params.keys()):
            # Convert indexed dict to list for lambda functions expecting list
            sorted_keys = sorted(params.keys(), key=int)
            param_list = [params[k] for k in sorted_keys]
            return self.func(param_list)
        return self.func(params)


class ParameterOptimizer:
    """Simple evolutionary parameter optimizer."""

    def __init__(self, objective: ObjectiveFunction, population_size: int = 20, generations: int = 50, mutation_rate: float = 0.1, seed: int = 42, max_history: int = 1000):
        self.objective = objective
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)
        self.best_params: Optional[Dict[str, float]] = None
        self.best_score: float = float('inf')
        self.history: deque = deque(maxlen=max_history)

    def _random_params(self) -> Dict[str, float]:
        params = {}
        for name, (lo, hi) in self.objective.param_bounds.items():
            params[name] = self.rng.uniform(lo, hi)
        return params

    def _mutate(self, params: Dict[str, float], sigma: float = None) -> Dict[str, float]:
        if sigma is None:
            sigma = self.mutation_rate
        new_params = {}
        for name, val in params.items():
            lo, hi = self.objective.param_bounds[name]
            new_val = val + self.rng.gauss(0, sigma * (hi - lo))
            new_params[name] = max(lo, min(hi, new_val))
        return new_params

    def optimize(self, generations: int = None) -> Dict[str, float]:
        if generations is None:
            generations = self.generations
        population = [self._random_params() for _ in range(self.population_size)]

        for gen in range(generations):
            scored = [(p, self.objective.evaluate(p)) for p in population]
            scored.sort(key=lambda x: x[1])

            if scored[0][1] < self.best_score:
                self.best_score = scored[0][1]
                self.best_params = scored[0][0].copy()

            self.history.append({"generation": gen, "best_score": round(self.best_score, 6)})

            # keep top half, mutate to fill
            survivors = [p for p, _ in scored[:self.population_size // 2]]
            children = [self._mutate(self.rng.choice(survivors)) for _ in range(self.population_size // 2)]
            population = survivors + children

        return self.best_params or {}

    def get_result(self) -> Dict:
        return {"best_params": self.best_params, "best_score": round(self.best_score, 6)}

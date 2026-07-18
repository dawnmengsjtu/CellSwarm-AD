# -*- coding: utf-8 -*-
"""2D multi-species reaction-diffusion grid using NumPy vectorization."""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass
class MultiSpeciesGrid:
    """2D grid supporting multiple diffusing species.

    Uses NumPy vectorized Laplacian (np.roll) for fast diffusion.
    """
    width: int = 100
    height: int = 100
    species: Dict[str, np.ndarray] = field(default_factory=dict)
    diffusion_rates: Dict[str, float] = field(default_factory=dict)
    decay_rates: Dict[str, float] = field(default_factory=dict)

    def add_species(self, name: str, diffusion_rate: float = 0.1, decay_rate: float = 0.01) -> None:
        """Add a new species to the grid."""
        self.species[name] = np.zeros((self.height, self.width))
        self.diffusion_rates[name] = diffusion_rate
        self.decay_rates[name] = decay_rate

    def add_source(self, name: str, x: int, y: int, amount: float) -> None:
        """Add concentration at a point."""
        if name not in self.species:
            raise KeyError(f"Species '{name}' not found")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Position ({x}, {y}) out of bounds")
        self.species[name][y, x] += amount

    def get_concentration(self, name: str, x: int, y: int) -> float:
        """Get concentration at a point."""
        if name not in self.species:
            raise KeyError(f"Species '{name}' not found")
        return float(self.species[name][y % self.height, x % self.width])

    def get_gradient(self, name: str, x: int, y: int) -> Tuple[float, float]:
        """Get concentration gradient at a point (central difference)."""
        if name not in self.species:
            raise KeyError(f"Species '{name}' not found")
        grid = self.species[name]
        h, w = self.height, self.width
        dx = (grid[y % h, (x + 1) % w] - grid[y % h, (x - 1) % w]) / 2.0
        dy = (grid[(y + 1) % h, x % w] - grid[(y - 1) % h, x % w]) / 2.0
        return (float(dx), float(dy))

    def step(self, dt: float = 0.01) -> None:
        """Advance all species by one timestep using vectorized diffusion."""
        for name in list(self.species.keys()):
            grid = self.species[name]
            D = self.diffusion_rates[name]
            decay = self.decay_rates[name]

            # Vectorized 4-neighbor Laplacian using np.roll
            laplacian = (
                np.roll(grid, 1, axis=0) +
                np.roll(grid, -1, axis=0) +
                np.roll(grid, 1, axis=1) +
                np.roll(grid, -1, axis=1) -
                4.0 * grid
            )

            grid += D * laplacian * dt - decay * grid * dt
            np.maximum(grid, 0.0, out=grid)
            self.species[name] = grid

    def total(self, name: str) -> float:
        """Get total concentration of a species."""
        if name not in self.species:
            raise KeyError(f"Species '{name}' not found")
        return float(np.sum(self.species[name]))

    def mean_concentration(self, name: str) -> float:
        """Get mean concentration of a species."""
        if name not in self.species:
            raise KeyError(f"Species '{name}' not found")
        return float(np.mean(self.species[name]))

# -*- coding: utf-8 -*-
"""Cell migration: random walk and chemotaxis."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .grid_2d import MultiSpeciesGrid


@dataclass
class CellMigration:
    """Cell migration in 2D space."""
    width: int = 100
    height: int = 100

    def random_walk(self, positions: List[Tuple[int, int]], step_size: float = 1.0, seed: int = 42) -> List[Tuple[int, int]]:
        """Brownian motion random walk."""
        rng = np.random.default_rng(seed)
        new_positions = []
        for x, y in positions:
            dx = int(round(rng.normal(0, step_size)))
            dy = int(round(rng.normal(0, step_size)))
            nx = max(0, min(self.width - 1, x + dx))
            ny = max(0, min(self.height - 1, y + dy))
            new_positions.append((nx, ny))
        return new_positions

    def chemotaxis(self, positions: List[Tuple[int, int]], grid: MultiSpeciesGrid, species_name: str, strength: float = 1.0) -> List[Tuple[int, int]]:
        """Move cells along concentration gradient (chemotaxis).

        Microglia migrate toward Aβ deposits.
        """
        new_positions = []
        for x, y in positions:
            gx, gy = grid.get_gradient(species_name, x, y)
            mag = (gx ** 2 + gy ** 2) ** 0.5
            if mag > 1e-10:
                dx = int(round(strength * gx / mag))
                dy = int(round(strength * gy / mag))
            else:
                dx, dy = 0, 0
            nx = max(0, min(self.width - 1, x + dx))
            ny = max(0, min(self.height - 1, y + dy))
            new_positions.append((nx, ny))
        return new_positions

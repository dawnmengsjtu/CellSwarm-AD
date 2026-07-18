# -*- coding: utf-8 -*-
"""Diffusion grid for molecular species in tissue."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DiffusionGrid:
    """2D diffusion grid for a single molecular species."""
    width: int = 50
    height: int = 50
    diffusion_rate: float = 0.1
    decay_rate: float = 0.01
    grid: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.grid:
            self.grid = [[0.0] * self.width for _ in range(self.height)]

    def step(self, dt: float = 0.01) -> None:
        new_grid = [[0.0] * self.width for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        neighbors.append(self.grid[ny][nx])
                avg_neighbor = sum(neighbors) / len(neighbors) if neighbors else 0
                diffusion = self.diffusion_rate * (avg_neighbor - self.grid[y][x]) * dt
                decay = self.decay_rate * self.grid[y][x] * dt
                new_grid[y][x] = max(0.0, self.grid[y][x] + diffusion - decay)
        self.grid = new_grid

    def get_value(self, x: int, y: int) -> float:
        return self.grid[y % self.height][x % self.width]

    def set_value(self, x: int, y: int, val: float) -> None:
        self.grid[y % self.height][x % self.width] = val

    def total(self) -> float:
        return sum(sum(row) for row in self.grid)

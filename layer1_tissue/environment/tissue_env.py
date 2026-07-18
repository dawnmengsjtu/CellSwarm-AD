# -*- coding: utf-8 -*-
"""Tissue-level environment managing spatial layout and diffusion."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
import math
from .diffusion import DiffusionGrid


@dataclass
class TissueEnvironment:
    """Manages the tissue microenvironment for cell agents."""
    width: int = 100
    height: int = 100
    cell_positions: Dict[str, tuple] = field(default_factory=dict)
    abeta_grid: Optional[DiffusionGrid] = None
    time: float = 0.0

    def __post_init__(self):
        if self.abeta_grid is None:
            self.abeta_grid = DiffusionGrid(
                width=self.width,
                height=self.height,
                diffusion_rate=0.1,
                decay_rate=0.01
            )

    @property
    def abeta_field(self) -> List[List[float]]:
        """Get the Abeta concentration field."""
        return self.abeta_grid.grid

    def place_cell(self, cell_id: str, pos: Tuple[int, int]) -> None:
        """Place a cell at position (x, y)."""
        x, y = pos
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise ValueError(f"Position ({x}, {y}) out of bounds (0-{self.width}, 0-{self.height})")
        self.cell_positions[cell_id] = (x, y)

    def get_local_abeta(self, location: Union[str, Tuple[int, int]]) -> float:
        """Get local Abeta concentration.

        Args:
            location: Either a cell_id (str) or (x, y) tuple

        Returns:
            Local Abeta concentration

        Raises:
            ValueError: If cell_id not found or coordinates out of bounds
        """
        if isinstance(location, str):
            # cell_id
            pos = self.cell_positions.get(location)
            if pos is None:
                raise ValueError(f"Cell {location} not found in tissue")
            x, y = pos
        else:
            # (x, y) tuple
            x, y = location
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                raise ValueError(f"Position ({x}, {y}) out of bounds")
        return self.abeta_grid.get_value(x, y)

    def deposit_abeta(self, pos: Tuple[int, int], amount: float) -> None:
        """Deposit Abeta at position (x, y).

        Args:
            pos: (x, y) coordinates
            amount: Amount to deposit (must be non-negative)

        Raises:
            ValueError: If amount is negative or position out of bounds
        """
        if amount < 0:
            raise ValueError(f"Amount must be non-negative, got {amount}")
        x, y = pos
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise ValueError(f"Position ({x}, {y}) out of bounds")
        current = self.abeta_grid.get_value(x, y)
        self.abeta_grid.set_value(x, y, current + amount)

    def step(self, dt: float = 0.01) -> None:
        """Advance one time step with diffusion.

        Args:
            dt: Timestep size (must be positive)

        Raises:
            ValueError: If dt is non-positive
        """
        if dt <= 0:
            raise ValueError(f"Timestep must be positive, got {dt}")
        self.time += dt
        self.abeta_grid.step(dt)

    def get_state(self) -> Dict:
        total_abeta = self.abeta_grid.total()
        return {
            "time": round(self.time, 4),
            "n_cells": len(self.cell_positions),
            "total_abeta": round(total_abeta, 4),
            "grid_size": (self.width, self.height),
        }

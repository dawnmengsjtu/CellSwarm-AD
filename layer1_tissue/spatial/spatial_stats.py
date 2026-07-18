# -*- coding: utf-8 -*-
"""Spatial statistics for cell distributions."""
from typing import List, Tuple
import numpy as np


class SpatialStats:
    """Spatial statistics for analyzing cell distributions."""

    @staticmethod
    def nearest_neighbor_distances(positions: List[Tuple[int, int]]) -> np.ndarray:
        """Compute nearest neighbor distance for each cell."""
        if len(positions) < 2:
            return np.array([0.0])
        coords = np.array(positions, dtype=float)
        n = len(coords)
        distances = np.full(n, np.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    if d < distances[i]:
                        distances[i] = d
        return distances

    @staticmethod
    def clustering_index(positions: List[Tuple[int, int]], width: int, height: int) -> float:
        """Compute clustering index (ratio of observed to expected NN distance).

        < 1.0 = clustered, ~1.0 = random, > 1.0 = dispersed
        """
        if len(positions) < 2:
            return 1.0
        nn_dists = SpatialStats.nearest_neighbor_distances(positions)
        observed_mean = float(np.mean(nn_dists))
        # Expected NN distance for random Poisson process
        density = len(positions) / (width * height)
        expected_mean = 0.5 / np.sqrt(density) if density > 0 else 1.0
        return observed_mean / expected_mean if expected_mean > 0 else 1.0

    @staticmethod
    def mean_concentration(grid_array: np.ndarray) -> float:
        """Mean concentration of a 2D grid."""
        return float(np.mean(grid_array))

    @staticmethod
    def hotspot_count(grid_array: np.ndarray, threshold: float) -> int:
        """Count grid cells above a concentration threshold."""
        return int(np.sum(grid_array > threshold))

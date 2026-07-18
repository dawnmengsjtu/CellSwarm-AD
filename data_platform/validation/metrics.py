# -*- coding: utf-8 -*-
"""Validation metrics for model-experiment comparison."""
import math
from typing import List


def rmse(predicted: List[float], observed: List[float]) -> float:
    """Root Mean Square Error."""
    if len(predicted) != len(observed):
        raise ValueError("Lists must have same length")
    if len(predicted) == 0:
        raise ValueError("Lists must not be empty")
    mse = sum((p - o) ** 2 for p, o in zip(predicted, observed)) / len(predicted)
    return math.sqrt(mse)


def r_squared(predicted: List[float], observed: List[float]) -> float:
    """Coefficient of determination (R²)."""
    if len(predicted) != len(observed):
        raise ValueError("Lists must have same length")
    if len(predicted) == 0:
        raise ValueError("Lists must not be empty")
    mean_obs = sum(observed) / len(observed)
    ss_res = sum((o - p) ** 2 for p, o in zip(predicted, observed))
    ss_tot = sum((o - mean_obs) ** 2 for o in observed)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def mean_absolute_error(predicted: List[float], observed: List[float]) -> float:
    """Mean Absolute Error."""
    if len(predicted) != len(observed):
        raise ValueError("Lists must have same length")
    if len(predicted) == 0:
        raise ValueError("Lists must not be empty")
    return sum(abs(p - o) for p, o in zip(predicted, observed)) / len(predicted)


def normalized_rmse(predicted: List[float], observed: List[float]) -> float:
    """RMSE normalized by observed range."""
    if len(predicted) != len(observed):
        raise ValueError("Lists must have same length")
    obs_range = max(observed) - min(observed)
    if obs_range == 0:
        return 0.0
    return rmse(predicted, observed) / obs_range

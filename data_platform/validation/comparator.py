# -*- coding: utf-8 -*-
"""Model vs experiment comparator."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .metrics import rmse, r_squared, mean_absolute_error, normalized_rmse


@dataclass
class ComparisonResult:
    """Result of model-experiment comparison."""
    metric_name: str
    rmse: float
    r_squared: float
    mae: float
    nrmse: float
    n_points: int
    passed: bool = False
    threshold: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric_name,
            "rmse": round(self.rmse, 6),
            "r_squared": round(self.r_squared, 6),
            "mae": round(self.mae, 6),
            "nrmse": round(self.nrmse, 6),
            "n_points": self.n_points,
            "passed": self.passed,
        }


class ModelComparator:
    """Compare model output against experimental data."""

    def __init__(self, rmse_threshold: float = 0.1, r2_threshold: float = 0.8):
        self.rmse_threshold = rmse_threshold
        self.r2_threshold = r2_threshold
        self.results: List[ComparisonResult] = []

    def compare(self, name: str, predicted: List[float], observed: List[float]) -> ComparisonResult:
        """Compare predicted vs observed values.

        Args:
            name: Name of the metric being compared
            predicted: Model predictions
            observed: Experimental observations

        Returns:
            ComparisonResult with all metrics
        """
        r = ComparisonResult(
            metric_name=name,
            rmse=rmse(predicted, observed),
            r_squared=r_squared(predicted, observed),
            mae=mean_absolute_error(predicted, observed),
            nrmse=normalized_rmse(predicted, observed),
            n_points=len(predicted),
            threshold=self.rmse_threshold,
        )
        r.passed = r.rmse <= self.rmse_threshold and r.r_squared >= self.r2_threshold
        self.results.append(r)
        return r

    def summary(self) -> Dict:
        """Get summary of all comparisons."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "avg_r2": sum(r.r_squared for r in self.results) / len(self.results),
            "avg_rmse": sum(r.rmse for r in self.results) / len(self.results),
        }

# -*- coding: utf-8 -*-
"""Parameter database schema."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Parameter:
    """Scientific parameter with source and confidence."""
    name: str
    value: float
    unit: str
    source: str                    # Literature citation
    confidence: float = 1.0        # 0-1 confidence score
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    category: str = "general"      # abeta, tau, calcium, microglia, neuron
    notes: str = ""

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(f"min_value ({self.min_value}) > max_value ({self.max_value})")
            if not self.min_value <= self.value <= self.max_value:
                raise ValueError(f"value ({self.value}) not in range [{self.min_value}, {self.max_value}]")

    def in_range(self, test_value: float) -> bool:
        """Check if a value is within the parameter's range."""
        if self.min_value is None or self.max_value is None:
            return True
        return self.min_value <= test_value <= self.max_value

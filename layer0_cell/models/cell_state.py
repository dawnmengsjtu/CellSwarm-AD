# -*- coding: utf-8 -*-
"""Cell state transition model."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CellStateModel:
    """Tracks and predicts cell state transitions."""
    state_history: List[Dict] = field(default_factory=list)

    def record(self, state: Dict) -> None:
        self.state_history.append(state)

    def predict_next(self, current_state: Dict = None) -> Dict:
        """Simple linear extrapolation of numeric fields."""
        if current_state is None and len(self.state_history) >= 1:
            current_state = self.state_history[-1]
        if current_state is None:
            return {}

        predicted = dict(current_state)
        if len(self.state_history) >= 2:
            prev = self.state_history[-2]
            curr = self.state_history[-1]
            for k in curr:
                if isinstance(curr[k], (int, float)) and k in prev and isinstance(prev[k], (int, float)):
                    delta = curr[k] - prev[k]
                    predicted[k] = curr[k] + delta
        return predicted

    def get_trajectory(self, key: str = None) -> List:
        """Get trajectory of a specific key, or all states if key is None."""
        if key is None:
            return self.state_history
        return [s[key] for s in self.state_history if key in s]

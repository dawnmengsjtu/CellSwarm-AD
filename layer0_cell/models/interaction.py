# -*- coding: utf-8 -*-
"""Cell-cell interaction model."""
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class InteractionModel:
    """Models interactions between cell agents."""
    interaction_strength: float = 0.5

    def compute_interaction(self, source_state, target_state) -> Dict:
        """Compute interaction effect from source to target.

        Args:
            source_state: Either an agent object or a state dict
            target_state: Either an agent object or a state dict
        """
        # Convert agents to states if needed
        if hasattr(source_state, 'get_state'):
            source_state = source_state.get_state()
        if hasattr(target_state, 'get_state'):
            target_state = target_state.get_state()

        effect = {}
        cytokine = source_state.get("cytokine_production", 0.0)
        # Handle both dict and float cytokine_production
        if isinstance(cytokine, dict):
            cytokine = sum(cytokine.values())
        effect["cytokine_input"] = cytokine * self.interaction_strength
        effect["calcium_input"] = source_state.get("calcium_wave", 0.0) * self.interaction_strength
        return effect

    def pairwise_interactions(self, agents: List) -> List[Tuple[int, int, Dict]]:
        """Compute pairwise interactions between agents."""
        states = [a.get_state() if hasattr(a, 'get_state') else a for a in agents]
        results = []
        for i in range(len(states)):
            for j in range(len(states)):
                if i != j:
                    effect = self.compute_interaction(states[i], states[j])
                    results.append((i, j, effect))
        return results

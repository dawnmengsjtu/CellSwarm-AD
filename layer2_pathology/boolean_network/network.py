# -*- coding: utf-8 -*-
"""Boolean gene regulatory network for AD pathways."""
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional


@dataclass
class GeneNode:
    """A node in the boolean gene network."""
    name: str
    state: bool = False
    update_rule: Optional[Callable] = None

    def update(self, network_state: Dict[str, bool]) -> bool:
        if self.update_rule:
            self.state = self.update_rule(network_state)
        return self.state


class BooleanGeneNetwork:
    """Boolean network modeling gene regulatory interactions in AD."""

    def __init__(self):
        self.nodes: Dict[str, GeneNode] = {}
        self.history: List[Dict[str, bool]] = []

    def add_node(self, node: GeneNode) -> None:
        self.nodes[node.name] = node

    def get_state(self, gene_name: str = None) -> Dict[str, bool]:
        """Get state of a specific gene or all genes."""
        if gene_name is not None:
            return self.nodes[gene_name].state if gene_name in self.nodes else None
        return {name: node.state for name, node in self.nodes.items()}

    def step(self) -> Dict[str, bool]:
        current = self.get_state()
        self.history.append(current)
        for node in self.nodes.values():
            node.update(current)
        return self.get_state()

    def run(self, steps: int = 10) -> List[Dict[str, bool]]:
        results = []
        for _ in range(steps):
            results.append(self.step())
        return results

    def set_state(self, name: str, state: bool) -> None:
        if name in self.nodes:
            self.nodes[name].state = state

    @classmethod
    def build_ad_network(cls) -> 'BooleanGeneNetwork':
        """Build a default AD-related gene network."""
        bn = cls()
        bn.add_node(GeneNode("APP", True, lambda s: True))
        bn.add_node(GeneNode("BACE1", False, lambda s: s.get("APP", False)))
        bn.add_node(GeneNode("ABETA", False, lambda s: s.get("APP", False) and s.get("BACE1", False)))
        bn.add_node(GeneNode("NFKB", False, lambda s: s.get("ABETA", False)))
        bn.add_node(GeneNode("TNF_ALPHA", False, lambda s: s.get("NFKB", False)))
        bn.add_node(GeneNode("CASP3", False, lambda s: s.get("TNF_ALPHA", False) or s.get("ABETA", False)))
        bn.add_node(GeneNode("APOPTOSIS", False, lambda s: s.get("CASP3", False)))
        bn.add_node(GeneNode("TREM2", True, lambda s: not s.get("ABETA", False)))
        bn.add_node(GeneNode("PHAGOCYTOSIS", False, lambda s: s.get("TREM2", False) and not s.get("APOPTOSIS", False)))
        return bn

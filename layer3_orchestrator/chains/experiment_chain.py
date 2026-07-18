# -*- coding: utf-8 -*-
"""Experiment design chain for AD simulations."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..llm_interface import LLMWrapper


@dataclass
class ExperimentSpec:
    """Specification for a simulation experiment."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: int = 1000
    metrics: List[str] = field(default_factory=list)
    hypothesis: str = ""


class ExperimentChain:
    """Uses LLM to design and refine simulation experiments."""

    def __init__(self, llm: LLMWrapper):
        self.llm = llm
        self.experiments: List[ExperimentSpec] = []

    def design_experiment(self, hypothesis: str, available_params: Optional[Dict] = None) -> ExperimentSpec:
        """Design an experiment to test a hypothesis."""
        params_str = str(available_params) if available_params else "default AD simulation parameters"
        prompt = (
            f"Design a simulation experiment to test this hypothesis:\n"
            f"Hypothesis: {hypothesis}\n"
            f"Available parameters: {params_str}\n"
            f"Specify: experiment name, parameter ranges, duration, and metrics to track."
        )
        response = self.llm.generate(prompt, system_prompt="You are an AD computational experiment designer.")
        spec = ExperimentSpec(
            name="abeta_sweep_experiment",
            description=response,
            parameters=available_params or {"abeta_production_rate": [0.05, 0.1, 0.2, 0.5]},
            duration=1000,
            metrics=["abeta_total", "microglial_activation", "tau_phosphorylation", "neuron_viability"],
            hypothesis=hypothesis,
        )
        self.experiments.append(spec)
        return spec

    def refine_experiment(self, spec: ExperimentSpec, results: Dict) -> ExperimentSpec:
        """Refine an experiment based on previous results."""
        prompt = (
            f"Refine this experiment based on results:\n"
            f"Original: {spec.name} — {spec.description[:200]}\n"
            f"Results: {results}\n"
            f"Suggest refined parameters and metrics."
        )
        response = self.llm.generate(prompt)
        refined = ExperimentSpec(
            name=f"{spec.name}_refined",
            description=response,
            parameters=spec.parameters,
            duration=spec.duration,
            metrics=spec.metrics,
            hypothesis=spec.hypothesis,
        )
        self.experiments.append(refined)
        return refined

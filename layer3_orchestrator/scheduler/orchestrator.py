# -*- coding: utf-8 -*-
"""Top-level orchestrator coordinating all layers."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import uuid

from ..llm_interface import LLMWrapper, LLMConfig
from ..chains import ReasoningChain, ExperimentChain, ReasoningType
from .task_queue import TaskQueue, Task


class Orchestrator:
    """Coordinates Layer 0-2 simulations via LLM-driven reasoning."""

    def __init__(self, llm: Optional[LLMWrapper] = None):
        self.llm = llm or LLMWrapper(LLMConfig(backend="mock"))
        self.reasoning = ReasoningChain(self.llm)
        self.experiment = ExperimentChain(self.llm)
        self.task_queue = TaskQueue()
        self.results: List[Dict] = []
        self._task_counter = 0

    def hypothesize(self, context: str) -> Dict:
        """Generate a hypothesis from simulation context."""
        result = self.reasoning.run(ReasoningType.HYPOTHESIS, context)
        return {"hypothesis": result.content, "confidence": result.confidence}

    def design(self, hypothesis: str, params: Optional[Dict] = None) -> Dict:
        """Design an experiment to test a hypothesis."""
        spec = self.experiment.design_experiment(hypothesis, params)
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "duration": spec.duration,
            "metrics": spec.metrics,
        }

    def schedule_task(self, name: str, fn: Callable[..., Any], **kwargs) -> Task:
        """Add a task to the queue."""
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, name=name, callable_fn=fn, kwargs=kwargs)
        self.task_queue.add(task)
        return task

    def run_pipeline(self, context: str) -> Dict:
        """Run full hypothesis → design → schedule pipeline."""
        hyp = self.hypothesize(context)
        design = self.design(hyp["hypothesis"])
        analysis = self.reasoning.run(ReasoningType.ANALYSIS, f"Hypothesis: {hyp['hypothesis']}, Design: {design['name']}")
        result = {
            "hypothesis": hyp,
            "experiment_design": design,
            "analysis": {"content": analysis.content, "confidence": analysis.confidence},
        }
        self.results.append(result)
        return result

    def get_status(self) -> Dict:
        return {
            "llm_stats": self.llm.get_stats(),
            "reasoning_history": len(self.reasoning.history),
            "experiments_designed": len(self.experiment.experiments),
            "task_queue": self.task_queue.get_status(),
            "pipeline_runs": len(self.results),
        }

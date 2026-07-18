# -*- coding: utf-8 -*-
"""Chain-of-thought reasoning for scientific discovery."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from ..llm_interface import LLMWrapper


class ReasoningType(Enum):
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CRITIQUE = "critique"


@dataclass
class ReasoningResult:
    """Result of a reasoning chain execution."""
    reasoning_type: ReasoningType
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ReasoningChain:
    """Multi-step reasoning chain for AD research."""

    PROMPTS = {
        ReasoningType.HYPOTHESIS: (
            "Based on the following AD simulation context, generate a testable hypothesis.\n"
            "Context: {context}\nFormulate a specific, mechanistic hypothesis:"
        ),
        ReasoningType.ANALYSIS: (
            "Analyze the following AD simulation results.\n"
            "Results: {context}\nProvide a detailed scientific analysis:"
        ),
        ReasoningType.SYNTHESIS: (
            "Synthesize the following findings into a coherent narrative.\n"
            "Findings: {context}\nSynthesis:"
        ),
        ReasoningType.CRITIQUE: (
            "Critically evaluate the following hypothesis and evidence.\n"
            "Content: {context}\nCritique:"
        ),
    }

    def __init__(self, llm: LLMWrapper):
        self.llm = llm
        self.history: List[ReasoningResult] = []

    def run(self, reasoning_type: ReasoningType, context: str) -> ReasoningResult:
        prompt_template = self.PROMPTS[reasoning_type]
        prompt = prompt_template.format(context=context)
        response = self.llm.generate(prompt, system_prompt="You are an Alzheimer's Disease research scientist.")
        result = ReasoningResult(
            reasoning_type=reasoning_type,
            content=response,
            confidence=0.75,
            evidence=[context[:200]],
            metadata={"prompt_length": len(prompt)},
        )
        self.history.append(result)
        return result

    def chain(self, steps: List[Dict]) -> List[ReasoningResult]:
        """Run multiple reasoning steps in sequence, feeding output forward."""
        results = []
        carry_context = ""
        for step in steps:
            rtype = step.get("type", ReasoningType.ANALYSIS)
            ctx = step.get("context", "") + carry_context
            result = self.run(rtype, ctx)
            results.append(result)
            carry_context = f"\nPrevious: {result.content[:300]}"
        return results

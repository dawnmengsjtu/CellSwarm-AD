# -*- coding: utf-8 -*-
"""Integration tests for Layer 3: Orchestrator."""
import pytest
from layer3_orchestrator import (
    LLMWrapper, LLMConfig,
    ReasoningChain, ReasoningType, ExperimentChain,
    Orchestrator, TaskQueue, Task
)


class TestLLMWrapper:
    def test_mock_backend(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        response = llm.generate("test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_call_history(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        llm.generate("prompt 1")
        llm.generate("prompt 2")
        assert len(llm.call_history) == 2

    def test_get_stats(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        llm.generate("test")
        stats = llm.get_stats()
        assert stats["backend"] == "mock"
        assert stats["total_calls"] == 1


class TestReasoningChain:
    def test_hypothesis_generation(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        chain = ReasoningChain(llm)
        result = chain.run(ReasoningType.HYPOTHESIS, "test context")
        assert result.reasoning_type == ReasoningType.HYPOTHESIS
        assert len(result.content) > 0
        assert 0 <= result.confidence <= 1

    def test_analysis(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        chain = ReasoningChain(llm)
        result = chain.run(ReasoningType.ANALYSIS, "test results")
        assert result.reasoning_type == ReasoningType.ANALYSIS
        assert len(result.content) > 0

    def test_chain_multiple_steps(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        chain = ReasoningChain(llm)
        steps = [
            {"type": ReasoningType.HYPOTHESIS, "context": "ctx1"},
            {"type": ReasoningType.ANALYSIS, "context": "ctx2"}
        ]
        results = chain.chain(steps)
        assert len(results) == 2
        assert len(chain.history) == 2


class TestExperimentChain:
    def test_design_experiment(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        chain = ExperimentChain(llm)
        spec = chain.design_experiment("test hypothesis")
        assert spec.name is not None
        assert len(spec.description) > 0
        assert isinstance(spec.parameters, dict)
        assert isinstance(spec.metrics, list)

    def test_refine_experiment(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        chain = ExperimentChain(llm)
        spec = chain.design_experiment("test hypothesis")
        refined = chain.refine_experiment(spec, {"result": "test"})
        assert refined.name.endswith("_refined")
        assert len(chain.experiments) == 2


class TestTaskQueue:
    def test_add_task(self):
        queue = TaskQueue()
        task = Task(task_id="t1", name="test", callable_fn=lambda: 42)
        queue.add(task)
        assert len(queue.tasks) == 1

    def test_next_task(self):
        queue = TaskQueue()
        task1 = Task(task_id="t1", name="test1", callable_fn=lambda: 1)
        task2 = Task(task_id="t2", name="test2", callable_fn=lambda: 2)
        queue.add(task1)
        queue.add(task2)
        next_task = queue.next()
        assert next_task.task_id == "t1"

    def test_run_next(self):
        queue = TaskQueue()
        task = Task(task_id="t1", name="test", callable_fn=lambda: 42)
        queue.add(task)
        result = queue.run_next()
        assert result.result == 42
        assert result.status.value == "completed"

    def test_run_all(self):
        queue = TaskQueue()
        queue.add(Task(task_id="t1", name="test1", callable_fn=lambda: 1))
        queue.add(Task(task_id="t2", name="test2", callable_fn=lambda: 2))
        results = queue.run_all()
        assert len(results) == 2
        assert all(r.status.value == "completed" for r in results)

    def test_get_status(self):
        queue = TaskQueue()
        queue.add(Task(task_id="t1", name="test", callable_fn=lambda: 1))
        status = queue.get_status()
        assert status["total"] == 1
        assert status["pending"] == 1


class TestOrchestrator:
    def test_hypothesize(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        orch = Orchestrator(llm)
        result = orch.hypothesize("test context")
        assert "hypothesis" in result
        assert "confidence" in result

    def test_design(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        orch = Orchestrator(llm)
        result = orch.design("test hypothesis")
        assert "name" in result
        assert "parameters" in result
        assert "metrics" in result

    def test_schedule_task(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        orch = Orchestrator(llm)
        task = orch.schedule_task("test", lambda: 42)
        assert task.name == "test"
        assert len(orch.task_queue.tasks) == 1

    def test_run_pipeline(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        orch = Orchestrator(llm)
        result = orch.run_pipeline("test context")
        assert "hypothesis" in result
        assert "experiment_design" in result
        assert "analysis" in result
        assert len(orch.results) == 1

    def test_get_status(self):
        llm = LLMWrapper(LLMConfig(backend="mock"))
        orch = Orchestrator(llm)
        orch.run_pipeline("test")
        status = orch.get_status()
        assert "llm_stats" in status
        assert "reasoning_history" in status
        assert "experiments_designed" in status
        assert "pipeline_runs" in status

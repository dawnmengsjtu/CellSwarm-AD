# -*- coding: utf-8 -*-
"""Integration tests for Layer 0: Cell agents, models, samplers."""
import pytest
from layer0_cell import (
    NeuronAgent, MicrogliaAgent, AstrocyteAgent,
    CellStateModel, InteractionModel,
    MonteCarloSampler, ImportanceSampler,
)
from layer0_cell.agents.microglia import ActivationState
from layer0_cell.agents.neuron import GENE_MAP


class TestNeuronAgent:
    def test_initial_state(self):
        n = NeuronAgent(agent_id="n1")
        assert n.viability == 1.0
        assert n.calcium > 0
        assert n.agent_id == "n1"

    def test_step_reduces_viability_under_abeta(self):
        n = NeuronAgent(agent_id="n1")
        for _ in range(100):
            n.step(dt=0.01, abeta_concentration=1.0)
        assert n.viability < 1.0

    def test_get_state_keys(self):
        n = NeuronAgent(agent_id="n1")
        state = n.get_state()
        assert "viability" in state
        assert "calcium" in state
        assert "agent_id" in state

    def test_gene_expression(self):
        n = NeuronAgent(agent_id="n1")
        state = n.get_state()
        assert "gene_expression" in state
        assert state["gene_expression"] == GENE_MAP

    def test_step_does_not_update_gene_weights(self):
        n = NeuronAgent(agent_id="n1")
        initial_weights = dict(n.gene_expression)
        for _ in range(100):
            n.step(dt=0.01, abeta_concentration=1.0)
        assert n.gene_expression == initial_weights == GENE_MAP


class TestMicrogliaAgent:
    def test_initial_m0(self):
        m = MicrogliaAgent(agent_id="m1")
        assert m.activation_state == ActivationState.M0

    def test_activation_under_abeta(self):
        m = MicrogliaAgent(agent_id="m1")
        # Run enough steps at high Aβ to push NF-κB well above 0.6
        # so state transition away from M0 is near-certain (>95%)
        for _ in range(500):
            m.step(dt=0.01, abeta_concentration=5.0)
        assert m.activation_state != ActivationState.M0

    def test_phagocytosis_rate(self):
        m = MicrogliaAgent(agent_id="m1")
        assert m.phagocytosis_rate >= 0

    def test_cytokine_production(self):
        m = MicrogliaAgent(agent_id="m1")
        assert isinstance(m.cytokine_production, dict)


class TestAstrocyteAgent:
    def test_initial_reactivity(self):
        a = AstrocyteAgent(agent_id="a1")
        assert a.reactivity >= 0

    def test_step_changes_state(self):
        a = AstrocyteAgent(agent_id="a1")
        initial = a.reactivity
        for _ in range(50):
            a.step(dt=0.01, abeta_concentration=1.0)
        assert a.reactivity != initial


class TestCellStateModel:
    def test_record_and_trajectory(self):
        model = CellStateModel()
        n = NeuronAgent(agent_id="n1")
        for _ in range(5):
            n.step(dt=0.01, abeta_concentration=0.1)
            model.record(n.get_state())
        traj = model.get_trajectory()
        assert len(traj) == 5

    def test_predict_next(self):
        model = CellStateModel()
        n = NeuronAgent(agent_id="n1")
        for _ in range(3):
            n.step(dt=0.01, abeta_concentration=0.1)
            model.record(n.get_state())
        pred = model.predict_next()
        assert pred is not None


class TestInteractionModel:
    def test_pairwise(self):
        agents = [NeuronAgent(agent_id="n1"), MicrogliaAgent(agent_id="m1")]
        model = InteractionModel()
        results = model.pairwise_interactions(agents)
        assert len(results) > 0

    def test_compute_interaction(self):
        n = NeuronAgent(agent_id="n1")
        m = MicrogliaAgent(agent_id="m1")
        model = InteractionModel()
        result = model.compute_interaction(n, m)
        assert isinstance(result, dict)


class TestSamplers:
    def test_monte_carlo_sample(self):
        mc = MonteCarloSampler(n_samples=50)
        samples = mc.sample({"x": (0.0, 1.0), "y": (-1.0, 1.0)})
        assert len(samples) == 50
        assert "x" in samples[0]

    def test_monte_carlo_evaluate(self):
        mc = MonteCarloSampler(n_samples=20)
        samples = mc.sample({"x": (0.0, 1.0)})
        results = mc.evaluate(samples, lambda s: s["x"] ** 2)
        assert len(results) == 20

    def test_importance_sampler(self):
        imp = ImportanceSampler(n_samples=30)
        weighted = imp.sample_with_weights({"x": (0.0, 1.0)}, lambda s: s["x"])
        assert len(weighted) == 30

    def test_importance_resample(self):
        imp = ImportanceSampler(n_samples=30)
        weighted = imp.sample_with_weights({"x": (0.0, 1.0)}, lambda s: max(s["x"], 0.01))
        resampled = imp.resample(weighted, n=10)
        assert len(resampled) == 10

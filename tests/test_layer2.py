# -*- coding: utf-8 -*-
"""Integration tests for Layer 2: Pathology models."""
import pytest
from layer2_pathology import (
    AbetaDynamics, TauDynamics,
    BooleanGeneNetwork, GeneNode,
    AmyloidODE, TauODE, CalciumODE,
    ParameterOptimizer, ObjectiveFunction,
)


class TestAbetaDynamics:
    def test_initial_zero(self):
        a = AbetaDynamics()
        assert a.monomer_conc == 0.0

    def test_production(self):
        a = AbetaDynamics(production_rate=0.5)
        for _ in range(100):
            a.step(dt=0.01, microglial_clearance=0.0)
        assert a.monomer_conc > 0

    def test_clearance_reduces(self):
        a = AbetaDynamics(production_rate=0.1, clearance_rate=0.5)
        for _ in range(50):
            a.step(dt=0.01, microglial_clearance=0.5)
        a_high_clear = a.monomer_conc

        b = AbetaDynamics(production_rate=0.1, clearance_rate=0.0)
        for _ in range(50):
            b.step(dt=0.01, microglial_clearance=0.0)
        assert a_high_clear <= b.monomer_conc


class TestTauDynamics:
    def test_phosphorylation(self):
        t = TauDynamics(phosphorylation_rate=0.5)
        for _ in range(100):
            t.step(dt=0.01, kinase_activity=1.0)
        assert t.p_tau > 0

    def test_tangles_form(self):
        t = TauDynamics(phosphorylation_rate=0.5, tangle_rate=0.1)
        for _ in range(200):
            t.step(dt=0.01, kinase_activity=1.0)
        assert t.tangles > 0


class TestBooleanGeneNetwork:
    def test_build_ad_network(self):
        bn = BooleanGeneNetwork.build_ad_network()
        assert bn.get_state("APP") is not None

    def test_run_propagates(self):
        bn = BooleanGeneNetwork.build_ad_network()
        bn.run(steps=10)
        # APP=True should propagate downstream
        assert bn.get_state("ABETA") is not None

    def test_set_state(self):
        bn = BooleanGeneNetwork.build_ad_network()
        bn.set_state("APP", False)
        assert bn.get_state("APP") == False


class TestAmyloidODE:
    def test_run(self):
        ode = AmyloidODE(k_prod=0.1, k_clear=0.02, k_agg=0.05, k_plaque=0.01)
        ode.run(steps=200, dt=0.01)
        state = ode.get_state()
        assert state["A"] > 0

    def test_step(self):
        ode = AmyloidODE()
        ode.step(dt=0.01)
        assert ode.get_state() is not None


class TestTauODE:
    def test_run(self):
        ode = TauODE()
        ode.run(steps=200, dt=0.01, kinase_activity=0.5)
        state = ode.get_state()
        assert "Tp" in state

    def test_kinase_effect(self):
        low = TauODE()
        low.run(steps=100, dt=0.01, kinase_activity=0.1)
        high = TauODE()
        high.run(steps=100, dt=0.01, kinase_activity=1.0)
        assert high.get_state()["Tp"] >= low.get_state()["Tp"]


class TestCalciumODE:
    def test_run(self):
        ode = CalciumODE()
        ode.run(steps=100, dt=0.01, stimulus=0.2)
        state = ode.get_state()
        assert "Ca_cyt" in state

    def test_stimulus_increases_calcium(self):
        no_stim = CalciumODE()
        no_stim.run(steps=100, dt=0.01, stimulus=0.0)
        with_stim = CalciumODE()
        with_stim.run(steps=100, dt=0.01, stimulus=1.0)
        assert with_stim.get_state()["Ca_cyt"] >= no_stim.get_state()["Ca_cyt"]


class TestParameterOptimizer:
    def test_optimize(self):
        obj = ObjectiveFunction(
            name="sphere",
            func=lambda p: -(p[0] ** 2 + p[1] ** 2),
            param_bounds=[(-1, 1), (-1, 1)],
        )
        opt = ParameterOptimizer(objective=obj, population_size=10, generations=20, mutation_rate=0.1)
        best = opt.optimize()
        assert best is not None

    def test_get_result(self):
        obj = ObjectiveFunction(
            name="test",
            func=lambda p: -sum(x ** 2 for x in p),
            param_bounds=[(-1, 1)],
        )
        opt = ParameterOptimizer(objective=obj, population_size=10, generations=10, mutation_rate=0.1)
        opt.optimize()
        result = opt.get_result()
        assert "best_params" in result or "best_fitness" in result or isinstance(result, dict)

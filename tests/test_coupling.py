# -*- coding: utf-8 -*-
"""Tests for Layer 2 coupling mechanisms."""
import pytest
from layer2_pathology.coupling import (
    AbetaCalciumCoupling,
    CalciumTauCoupling,
    TauNeuronCoupling,
    AbetaMicrogliaCoupling,
    MicrogliaNeuronCoupling,
    ADCascade,
)
from layer2_pathology.coupling.cascade import CascadeState


class TestAbetaCalciumCoupling:
    def test_zero_abeta_no_influx(self):
        c = AbetaCalciumCoupling()
        assert c.compute_influx(0.0) == 0.0

    def test_high_abeta_saturates(self):
        c = AbetaCalciumCoupling()
        low = c.compute_influx(0.01)
        high = c.compute_influx(100.0)
        assert high > low
        assert high < c.k_abeta_ca * 1.01  # approaches max

    def test_apply_increases_calcium(self):
        c = AbetaCalciumCoupling()
        ca = c.apply(0.1, 0.5, 0.01)
        assert ca > 0.1

    def test_negative_abeta_raises(self):
        c = AbetaCalciumCoupling()
        with pytest.raises(ValueError):
            c.compute_influx(-0.1)


class TestCalciumTauCoupling:
    def test_below_threshold_baseline(self):
        c = CalciumTauCoupling()
        rate = c.effective_phosphorylation_rate(0.1)
        assert rate == c.k_phos_base

    def test_above_threshold_increases(self):
        c = CalciumTauCoupling()
        rate = c.effective_phosphorylation_rate(0.5)
        assert rate > c.k_phos_base

    def test_apply_increases_ptau(self):
        c = CalciumTauCoupling()
        ptau = c.apply(0.0, 0.5, 0.01)
        assert ptau > 0.0

    def test_negative_calcium_raises(self):
        c = CalciumTauCoupling()
        with pytest.raises(ValueError):
            c.effective_phosphorylation_rate(-0.1)


class TestTauNeuronCoupling:
    def test_zero_ptau_no_damage(self):
        c = TauNeuronCoupling()
        assert c.compute_damage(0.0) == 0.0

    def test_high_ptau_high_damage(self):
        c = TauNeuronCoupling()
        d = c.compute_damage(10.0)
        assert d > 0.09  # close to k_damage

    def test_apply_reduces_viability(self):
        c = TauNeuronCoupling()
        v = c.apply(1.0, 0.5, 0.1)
        assert v < 1.0

    def test_viability_clamped(self):
        c = TauNeuronCoupling()
        v = c.apply(0.001, 10.0, 100.0)
        assert v >= 0.0


class TestAbetaMicrogliaCoupling:
    def test_zero_abeta_no_activation(self):
        c = AbetaMicrogliaCoupling()
        assert c.compute_activation(0.0) == 0.0

    def test_trem2_reduces_activation(self):
        c1 = AbetaMicrogliaCoupling(trem2_inhibition=0.0)
        c2 = AbetaMicrogliaCoupling(trem2_inhibition=0.5)
        assert c1.compute_activation(1.0) > c2.compute_activation(1.0)

    def test_apply_increases_nfkb(self):
        c = AbetaMicrogliaCoupling()
        nfkb = c.apply(0.0, 1.0, 0.1)
        assert nfkb > 0.0


class TestMicrogliaNeuronCoupling:
    def test_m1_damages(self):
        c = MicrogliaNeuronCoupling()
        v = c.apply(1.0, "M1", 0.1)
        assert v < 1.0

    def test_m2_protects(self):
        c = MicrogliaNeuronCoupling()
        effect = c.compute_net_effect(0.05, 0.6)
        assert effect < 0  # net protection

    def test_m0_neutral(self):
        c = MicrogliaNeuronCoupling()
        effect = c.compute_net_effect(0.1, 0.1)
        assert abs(effect) < 0.02  # roughly neutral

    def test_negative_tnf_raises(self):
        c = MicrogliaNeuronCoupling()
        with pytest.raises(ValueError):
            c.compute_net_effect(-0.1, 0.1)


class TestADCascade:
    def test_initial_state(self):
        cascade = ADCascade()
        state = cascade.get_state()
        assert state["viability"] == 1.0
        assert state["p_tau"] == 0.0

    def test_step_with_abeta(self):
        cascade = ADCascade()
        state = cascade.step(0.5, dt=0.01)
        assert state["calcium"] > 0.1
        assert state["p_tau"] > 0.0

    def test_high_abeta_reduces_viability(self):
        cascade = ADCascade()
        for _ in range(1000):
            cascade.step(1.0, dt=0.01)
        assert cascade.state.viability < 1.0

    def test_zero_abeta_stable(self):
        cascade = ADCascade()
        for _ in range(100):
            cascade.step(0.0, dt=0.01)
        assert cascade.state.viability > 0.99

    def test_run_returns_trajectory(self):
        cascade = ADCascade()
        traj = cascade.run(0.5, steps=50, dt=0.01)
        assert len(traj) == 50
        assert traj[-1]["time"] > traj[0]["time"]

    def test_reset(self):
        cascade = ADCascade()
        cascade.step(1.0, dt=0.1)
        cascade.reset()
        assert cascade.state.viability == 1.0
        assert cascade.state.p_tau == 0.0

    def test_microglia_polarization(self):
        cascade = ADCascade()
        # High Aβ should eventually activate microglia
        for _ in range(500):
            cascade.step(2.0, dt=0.01)
        assert cascade.state.microglia_state in ("M1", "M2")

    def test_negative_abeta_raises(self):
        cascade = ADCascade()
        with pytest.raises(ValueError):
            cascade.step(-0.1)

    def test_cascade_state_to_dict(self):
        s = CascadeState()
        d = s.to_dict()
        assert "viability" in d
        assert "calcium" in d
        assert "p_tau" in d
        assert "microglia_state" in d

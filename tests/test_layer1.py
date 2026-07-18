# -*- coding: utf-8 -*-
"""Integration tests for Layer 1: Tissue environment and communication."""
import pytest
from layer1_tissue import TissueEnvironment, DiffusionGrid, SignalBus, CellMessage


class TestTissueEnvironment:
    def test_create(self):
        env = TissueEnvironment(width=10, height=10)
        assert env.width == 10

    def test_place_and_get_abeta(self):
        env = TissueEnvironment(width=20, height=20)
        env.place_cell("n1", (5, 5))
        env.deposit_abeta((5, 5), 1.0)
        val = env.get_local_abeta((5, 5))
        assert val > 0

    def test_step_diffuses(self):
        env = TissueEnvironment(width=20, height=20)
        env.deposit_abeta((10, 10), 5.0)
        center_before = env.get_local_abeta((10, 10))
        for _ in range(20):
            env.step(dt=0.01)
        center_after = env.get_local_abeta((10, 10))
        neighbor = env.get_local_abeta((11, 10))
        # Diffusion should spread: center decreases, neighbor increases
        assert center_after < center_before or neighbor > 0

    def test_get_state(self):
        env = TissueEnvironment(width=10, height=10)
        state = env.get_state()
        assert isinstance(state, dict)


class TestDiffusionGrid:
    def test_set_get(self):
        g = DiffusionGrid(width=10, height=10, diffusion_rate=0.1, decay_rate=0.01)
        g.set_value(5, 5, 2.0)
        assert g.get_value(5, 5) == 2.0

    def test_diffusion_conserves_roughly(self):
        g = DiffusionGrid(width=10, height=10, diffusion_rate=0.05, decay_rate=0.0)
        g.set_value(5, 5, 10.0)
        initial_total = g.total()
        g.step(dt=0.01)
        # With no decay, total should be roughly conserved
        assert abs(g.total() - initial_total) < initial_total * 0.5

    def test_decay_reduces_total(self):
        g = DiffusionGrid(width=10, height=10, diffusion_rate=0.0, decay_rate=0.5)
        g.set_value(5, 5, 10.0)
        g.step(dt=0.1)
        assert g.total() < 10.0


class TestSignalBus:
    def test_send_and_receive(self):
        bus = SignalBus()
        msg = CellMessage(sender_id="m1", receiver_id="n1", signal_type="cytokine", payload={"tnf": 0.5}, timestamp=0.0)
        bus.send(msg)
        msgs = bus.get_messages_for("n1")
        assert len(msgs) == 1
        assert msgs[0].payload["tnf"] == 0.5

    def test_broadcast(self):
        bus = SignalBus()
        bus.subscribe("n1", lambda msg: None)
        bus.subscribe("n2", lambda msg: None)
        bus.broadcast(sender_id="m1", signal_type="alarm", payload={"level": 1})
        assert len(bus.get_messages_for("n1")) >= 1
        assert len(bus.get_messages_for("n2")) >= 1

    def test_clear(self):
        bus = SignalBus()
        bus.send(CellMessage(sender_id="a", receiver_id="b", signal_type="x", payload={}, timestamp=0.0))
        bus.clear()
        assert len(bus.get_messages_for("b")) == 0

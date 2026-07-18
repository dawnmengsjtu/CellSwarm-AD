import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("run_validation.py")
SPEC = importlib.util.spec_from_file_location("mechanism_validation", MODULE_PATH)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_combination_is_donepezil_memantine():
    combo = mod.arms()["donepezil_memantine"]
    assert len(combo.drugs) == 2
    assert mod.drug_effect(combo, True) > 0
    assert mod.drug_effect(combo, False) == 0


def test_hc3_recovers_exact_interaction():
    trt = np.tile([0.0, 1.0], 20)
    genotype = np.repeat([0.0, 1.0, 2.0, 1.0], 10)
    x = np.column_stack([np.ones(40), trt, genotype, trt * genotype])
    y = x @ np.array([1.0, 2.0, 3.0, 4.0]) + np.linspace(-0.01, 0.01, 40)
    result = mod.hc3_ols(y, x, ["i", "t", "g", "txg"])
    assert abs(result[3]["estimate"] - 4.0) < 0.02


def test_grid_uses_fixed_physical_geometry_and_stable_cfl():
    _, a = mod.simulate_grid(50, final_time=0.001)
    _, b = mod.simulate_grid(100, final_time=0.001)
    assert b["dx"] == a["dx"] / 2
    assert a["source_radius"] == b["source_radius"]
    assert a["roi_radius"] == b["roi_radius"]
    assert a["diffusion_cfl"] <= 0.2 + 1e-12
    assert b["diffusion_cfl"] <= 0.2 + 1e-12
    assert abs(a["represented_source_area"] - b["represented_source_area"]) < 0.003


def test_grid_refinement_reduces_l2_error():
    f50, _ = mod.simulate_grid(50, final_time=0.001)
    f100, _ = mod.simulate_grid(100, final_time=0.001)
    f200, _ = mod.simulate_grid(200, final_time=0.001)
    ref50 = f200.reshape(50, 4, 50, 4).mean(axis=(1, 3))
    ref100 = f200.reshape(100, 2, 100, 2).mean(axis=(1, 3))
    assert np.linalg.norm(f100 - ref100) / np.linalg.norm(ref100) < np.linalg.norm(f50 - ref50) / np.linalg.norm(ref50)

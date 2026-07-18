#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CellSwarm-AD architecture demo across the four software layers."""

import yaml
import os

# ���� Layer 0: Cell ����
from layer0_cell import (
    NeuronAgent, MicrogliaAgent, AstrocyteAgent,
    CellStateModel, InteractionModel,
    MonteCarloSampler, ImportanceSampler,
)

# ���� Layer 1: Tissue ����
from layer1_tissue import TissueEnvironment, DiffusionGrid, SignalBus, CellMessage

# ���� Layer 2: Pathology ����
from layer2_pathology import (
    AbetaDynamics, TauDynamics,
    BooleanGeneNetwork,
    AmyloidODE, TauODE, CalciumODE,
    ParameterOptimizer, ObjectiveFunction,
)

# ���� Layer 3: Orchestrator ����
from layer3_orchestrator import (
    LLMWrapper, LLMConfig,
    ReasoningChain, ReasoningType,
    ExperimentChain,
    Orchestrator,
)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_demo():
    print("=" * 60)
    print("  CellSwarm-AD - Multi-Layer AD Simulation Demo")
    print("=" * 60)

    # Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    cfg = load_config(cfg_path)
    dt = cfg["simulation"]["dt"]
    steps = cfg["simulation"]["steps"]
    print(f"\n[Config] dt={dt}, steps={steps}")

    # Ablation flags
    abl = cfg.get('ablation', {})
    enable_spatial = abl.get('enable_spatial', True)
    enable_inflammation = abl.get('enable_inflammation', True)
    enable_pkpd = abl.get('enable_pkpd', True)
    print(f"[Ablation] spatial={enable_spatial}, inflammation={enable_inflammation}, pkpd={enable_pkpd}")

    # Layer 0: Create cell agents
    print("\n[Layer 0: Cell Agents]")
    neuron = NeuronAgent(agent_id="neuron_0")
    microglia = MicrogliaAgent(agent_id="microglia_0")
    astrocyte = AstrocyteAgent(agent_id="astrocyte_0")

    state_model = CellStateModel()
    interaction_model = InteractionModel()

    for i in range(10):
        neuron.step(dt=dt, abeta_concentration=0.1 * (i + 1))
        microglia.step(dt=dt, abeta_concentration=0.1 * (i + 1))
        astrocyte.step(dt=dt, abeta_concentration=0.1 * (i + 1))
        state_model.record(neuron.get_state())

    print(f"  Neuron state: viability={neuron.viability:.4f}, calcium={neuron.calcium:.4f}")
    print(f"  Microglia: activation={microglia.activation_state.value}, nfkb={microglia.nfkb_activity:.4f}")
    print(f"  Astrocyte: reactivity={astrocyte.reactivity:.4f}")
    print(f"  State model trajectory length: {len(state_model.get_trajectory())}")

    # Interaction
    agents = [neuron, microglia, astrocyte]
    interactions = interaction_model.pairwise_interactions(agents)
    print(f"  Pairwise interactions computed: {len(interactions)}")

    # Samplers
    sampler_mc = MonteCarloSampler(n_samples=100)
    samples = sampler_mc.sample({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    print(f"  MC samples: {len(samples)}")

    sampler_is = ImportanceSampler(n_samples=50)
    weighted = sampler_is.sample_with_weights({"x": (0.0, 1.0)}, lambda s: s["x"])
    print(f"  IS weighted samples: {len(weighted)}")

    # Layer 1: Tissue Environment
    print("\n[Layer 1: Tissue Environment]")
    if enable_spatial:
        env_cfg = cfg["layer1_tissue"]["environment"]
        tissue = TissueEnvironment(width=env_cfg["width"], height=env_cfg["height"])
        tissue.place_cell("neuron_0", (25, 25))
        tissue.place_cell("microglia_0", (26, 25))
        tissue.deposit_abeta((25, 25), 0.5)

        for _ in range(5):
            tissue.step(dt=dt)

        local_abeta = tissue.get_local_abeta((25, 25))
        print(f"  Local Abeta at neuron: {local_abeta:.4f}")
        print(f"  Tissue state keys: {list(tissue.get_state().keys())}")
    else:
        print("  [ABLATION] Spatial diffusion disabled — skipping TissueEnvironment.")

    # Signal bus
    bus = SignalBus()
    bus.send(CellMessage(sender_id="microglia_0", receiver_id="neuron_0", signal_type="cytokine", payload={"tnf_alpha": 0.3}, timestamp=0.0))
    msgs = bus.get_messages_for("neuron_0")
    print(f"  Messages for neuron_0: {len(msgs)}")

    # Diffusion grid
    grid = DiffusionGrid(width=20, height=20, diffusion_rate=0.1, decay_rate=0.01)
    grid.set_value(10, 10, 1.0)
    for _ in range(10):
        grid.step(dt=dt)
    print(f"  Diffusion grid total after 10 steps: {grid.total():.4f}")

    # Layer 2: Pathology
    print("\n[Layer 2: Pathology]")
    abeta_cfg = cfg["layer2_pathology"]["abeta"]
    abeta = AbetaDynamics(
        production_rate=abeta_cfg["production_rate"],
        aggregation_rate=abeta_cfg["aggregation_rate"],
        clearance_rate=abeta_cfg["clearance_rate"],
    )
    tau_cfg = cfg["layer2_pathology"]["tau"]
    tau = TauDynamics(
        phosphorylation_rate=tau_cfg["phosphorylation_rate"],
        tangle_rate=tau_cfg["tangle_rate"],
        clearance_rate=tau_cfg["clearance_rate"],
    )

    for _ in range(50):
        abeta.step(dt=dt, microglial_clearance=microglia.phagocytosis_rate)
        tau.step(dt=dt, kinase_activity=0.5)

    print(f"  Abeta: monomer={abeta.monomer_conc:.4f}, oligomer={abeta.oligomer_conc:.4f}, plaque={abeta.plaque_conc:.4f}")
    print(f"  Tau: p_tau={tau.p_tau:.4f}, tangles={tau.tangles:.4f}")

    # Boolean network
    if enable_inflammation:
        bn = BooleanGeneNetwork.build_ad_network()
        bn.run(steps=5)
        print(f"  Boolean network states: { {n: bn.get_state(n) for n in ['APP', 'ABETA', 'NFKB', 'APOPTOSIS']} }")
    else:
        print("  [ABLATION] Inflammation (NF-κB Boolean network) disabled — skipping.")

    # ODE models
    ode_cfg = cfg["layer2_pathology"]["ode"]
    amyloid_ode = AmyloidODE(
        k_prod=ode_cfg["amyloid"]["k_prod"],
        k_clear=ode_cfg["amyloid"]["k_clear"],
        k_agg=ode_cfg["amyloid"]["k_agg"],
        k_plaque=ode_cfg["amyloid"]["k_plaque"],
    )
    tau_ode = TauODE()
    calcium_ode = CalciumODE(
        leak_rate=ode_cfg["calcium"]["leak_rate"],
        pump_rate=ode_cfg["calcium"]["pump_rate"],
        channel_rate=ode_cfg["calcium"]["channel_rate"],
    )

    amyloid_ode.run(steps=100, dt=dt)
    tau_ode.run(steps=100, dt=dt, kinase_activity=0.5)
    calcium_ode.run(steps=100, dt=dt, stimulus=0.2)

    print(f"  Amyloid ODE: {amyloid_ode.get_state()}")
    print(f"  Tau ODE: {tau_ode.get_state()}")
    print(f"  Calcium ODE: {calcium_ode.get_state()}")

    # Optimizer
    def simple_obj(params):
        return -((params[0] - 0.3) ** 2 + (params[1] - 0.7) ** 2)

    obj_fn = ObjectiveFunction(name="test_obj", func=simple_obj, param_bounds=[(0, 1), (0, 1)])
    optimizer = ParameterOptimizer(objective=obj_fn, population_size=20, generations=30, mutation_rate=0.1)
    best = optimizer.optimize()
    print(f"  Optimizer best: {optimizer.get_result()}")

    # Layer 3: Orchestrator
    print("\n[Layer 3: Orchestrator]")
    llm_cfg = cfg["layer3_orchestrator"]["llm"]
    llm = LLMWrapper(LLMConfig(backend=llm_cfg["backend"], model_name=llm_cfg["model_name"]))

    orchestrator = Orchestrator(llm=llm)

    context = (
        f"Neuron viability={neuron.viability:.3f}, "
        f"Abeta oligomer={abeta.oligomer_conc:.3f}, "
        f"Microglia activation={microglia.activation_state.value}, "
        f"Tau p_tau={tau.p_tau:.3f}"
    )

    pipeline_result = orchestrator.run_pipeline(context)
    print(f"  Hypothesis: {pipeline_result['hypothesis']['hypothesis'][:100]}...")
    print(f"  Experiment: {pipeline_result['experiment_design']['name']}")
    print(f"  Analysis: {pipeline_result['analysis']['content'][:100]}...")
    print(f"  Orchestrator status: {orchestrator.get_status()}")

    print("\n" + "=" * 60)
    print("  Architecture demo complete; quantitative modules are not claimed as end-to-end coupled.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

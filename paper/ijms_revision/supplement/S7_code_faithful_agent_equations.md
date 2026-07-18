# Supplementary Table S7. Code-faithful Layer 0 agent updates

This reconstruction describes the executable `step()` methods in `layer0_cell/agents/` as of 2026-07-16. It does not infer continuous-time biology beyond the implemented discrete updates. Let \(\Delta t\) denote `dt`, \(A\) Aβ concentration, and \(\operatorname{clamp}_{[a,b]}(x)=\max(a,\min(b,x))\). Unless stated otherwise, inputs and state variables are code-normalized and the source does not declare physical units.

## Neuron agent

Source: `layer0_cell/agents/neuron.py`, `NeuronAgent.step()`.

With calcium \(C\), tau phosphorylation \(T\), and viability \(V\), assignment order matters:

\[
C_{t+1}=C_t+0.05A_t\Delta t,
\]

\[
T_{t+1}=T_t+0.02C_{t+1}\Delta t,
\]

\[
V_{t+1}=\max\{0,V_t-0.1T_{t+1}\Delta t\}.
\]

There is no random term. Only viability has a lower clamp; calcium and tau phosphorylation are not clamped. The initial defaults are \(C_0=0.1\), \(T_0=0\), and \(V_0=1\). The 12 `GENE_MAP` values are returned in state but are not used by `step()`; they are model-assigned relative weights, not directly measured parameter values.

## Astrocyte agent

Source: `layer0_cell/agents/astrocyte.py`, `AstrocyteAgent.step()`.

Let \(R\) be reactivity, \(Y\) cytokine level, \(G\) glutamate uptake, and \(W\) calcium-wave output:

\[
R_{t+1}=\min\{1,R_t+0.15(Y_t+0.1A_t)\Delta t\},
\]

\[
G_{t+1}=\max\{0.1,0.8-0.5R_{t+1}\},\qquad W_{t+1}=0.3R_{t+1}.
\]

There is no random term and no lower clamp on reactivity. Defaults are \(R_0=0\), \(G_0=0.8\), and \(W_0=0\).

## Microglia agent

Source: `layer0_cell/agents/microglia.py`, `MicrogliaAgent.step()`.

Let \(N\) denote NF-κB activity. For an independent Gaussian draw \(\epsilon_N\sim\mathcal N(0,(0.01\Delta t)^2)\):

\[
N_{t+1}=\operatorname{clamp}_{[0,1]}\left[N_t+(0.1A_t-0.05N_t)\Delta t+\epsilon_N\right].
\]

An independent \(r\sim U(0,1)\) then redraws the activation state from the following piecewise distribution. The previous M0/M1/M2 state is not used in this draw.

| NF-κB interval | P(M1) | P(M2) | P(M0) |
|---|---:|---:|---:|
| \(N_{t+1}>0.6\) | 0.75 | 0.20 | 0.05 |
| \(0.3<N_{t+1}\leq0.6\) | 0.30 | 0.40 | 0.30 |
| \(0.1<N_{t+1}\leq0.3\) | 0.10 | 0.20 | 0.70 |
| \(N_{t+1}\leq0.1\) | 0 | 0.05 | 0.95 |

Functional outputs are state-dependent:

| State | Cytokine outputs | Raw phagocytosis \(q^*\) |
|---|---|---|
| M1 | TNF-α \(=0.8N\); IL-1β \(=0.6N\) | \(0.1+\epsilon_1\), \(\epsilon_1\sim\mathcal N(0,0.02^2)\) |
| M2 | IL-10 \(=0.3N\); TGF-β \(=0.2N\) | \(0.6+\epsilon_2\), \(\epsilon_2\sim\mathcal N(0,0.05^2)\) |
| M0 | empty cytokine dictionary | \(0.3+\epsilon_0\), \(\epsilon_0\sim\mathcal N(0,0.03^2)\) |

Finally, \(q_{t+1}=\operatorname{clamp}_{[0,1]}(q^*)\). Unlike the NF-κB noise, phagocytosis noise is not scaled by \(\Delta t\). No local random seed is set in this class.

## Oligodendrocyte agent

Source: `layer0_cell/agents/oligodendrocyte.py`, `OligodendrocyteAgent.step()`.

Let \(M\) be myelination capacity, \(I\) myelin integrity, \(P\) maturation progress, \(C\) calcium, and \(F\) TNF-α. With independent draws \(\epsilon_M,\epsilon_I\sim\mathcal N(0,(0.005\Delta t)^2)\) and \(\epsilon_P\sim\mathcal N(0,(0.003\Delta t)^2)\):

\[
M_{t+1}=\operatorname{clamp}_{[0,1]}(M_t-0.05A_t\Delta t-\epsilon_M),
\]

\[
I_{t+1}=\operatorname{clamp}_{[0,1]}(I_t-0.08C_t\Delta t-\epsilon_I),
\]

\[
s_t=1-\min(1,0.7F_t),
\]

\[
P_{t+1}=\operatorname{clamp}_{[0,1]}[P_t+0.02s_t\Delta t+\epsilon_P].
\]

The transition is unidirectional and evaluated after updating \(P\): OPC → pre-OL if \(P\geq0.4\); otherwise, if already pre-OL, pre-OL → mature-OL if \(P\geq0.8\). A single call cannot traverse both stages because the second condition is `elif`. There is no reverse transition.

## Endothelial agent

Source: `layer0_cell/agents/endothelial.py`, `EndothelialAgent.step()`.

Let \(J\) be tight-junction strength, \(B\) BBB integrity, \(F\) TNF-α, \(L\) IL-10, and \(Q\) net Aβ transport. With mutually independent \(\epsilon_J,\epsilon_B\sim\mathcal N(0,(0.005\Delta t)^2)\) and \(\epsilon_Q\sim\mathcal N(0,(0.01\Delta t)^2)\):

\[
J_{t+1}=\operatorname{clamp}_{[0,1]}\{J_t+[0.01(1-J_t)-0.04A_t]\Delta t+\epsilon_J\},
\]

\[
B_{t+1}=\operatorname{clamp}_{[0,1]}\{B_t+[0.05(J_{t+1}-B_t)+0.06L_t(1-B_t)-0.10F_t]\Delta t+\epsilon_B\},
\]

\[
Q_{t+1}=\operatorname{clamp}_{[-1,1]}[0.6B_{t+1}-0.3(1-B_{t+1})+\epsilon_Q].
\]

The transport update replaces the previous \(Q_t\); it is not an Euler increment. Positive transport is documented in code as clearance and negative transport as influx. Defaults are \(J_0=B_0=1\) and \(Q_0=0.5\).

## Implementation caveats

- `get_state()` rounds reported scalar states to four decimals, but internal updates retain full floating-point precision.
- Input validation differs by class: neuron, microglia, oligodendrocyte, and endothelial agents reject negative biological inputs and non-positive `dt`; astrocyte performs no such checks.
- Coefficients are labeled **heuristic/model-assigned** unless a direct parameter-level source is encoded. Biological citations in comments motivate mechanisms but do not establish the exact numerical weights.

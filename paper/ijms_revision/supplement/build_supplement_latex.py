"""Build the IJMS supplement as XeLaTeX source with code-derived tables."""
from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "latex"
OUT.mkdir(parents=True, exist_ok=True)
TEX = OUT / "IJMS_Supplementary_Materials.tex"


def esc(value):
    s = str(value)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


params = pd.read_csv(HERE / "selected_runtime_parameters.csv").fillna("")
cols = ["parameter", "value", "unit", "source_type"]

rows = []
for _, r in params.iterrows():
    parameter = esc(r["parameter"]).replace(r"\_", r"\_\allowbreak{}").replace(".", r".\allowbreak{}")
    vals = [parameter] + [esc(r[c]) for c in cols[1:]]
    rows.append(" & ".join(vals) + r" \\")

prompt_records = [
    ("S6-P1", "reasoning_chain.py / HYPOTHESIS",
     "Based on the following AD simulation context, generate a testable hypothesis. Context: {context} Formulate a specific, mechanistic hypothesis:",
     "You are an Alzheimer's Disease research scientist."),
    ("S6-P2", "reasoning_chain.py / ANALYSIS",
     "Analyze the following AD simulation results. Results: {context} Provide a detailed scientific analysis:",
     "You are an Alzheimer's Disease research scientist."),
    ("S6-P3", "reasoning_chain.py / SYNTHESIS",
     "Synthesize the following findings into a coherent narrative. Findings: {context} Synthesis:",
     "You are an Alzheimer's Disease research scientist."),
    ("S6-P4", "reasoning_chain.py / CRITIQUE",
     "Critically evaluate the following hypothesis and evidence. Content: {context} Critique:",
     "You are an Alzheimer's Disease research scientist."),
    ("S6-P5", "experiment_chain.py / design_experiment",
     "Design a simulation experiment to test this hypothesis: Hypothesis: {hypothesis} Available parameters: {params_str} Specify: experiment name, parameter ranges, duration, and metrics to track.",
     "You are an AD computational experiment designer."),
    ("S6-P6", "experiment_chain.py / refine_experiment",
     "Refine this experiment based on results: Original: {spec.name} -- {spec.description[:200]} Results: {results} Suggest refined parameters and metrics.",
     "No system prompt is supplied by this call."),
]

prompt_tex = []
for pid, source, user, system in prompt_records:
    prompt_tex.append(
        rf"\subsubsection*{{{esc(pid)} -- {esc(source)}}}" "\n"
        rf"\textbf{{User prompt template.}} \texttt{{\small {esc(user)}}}\par" "\n"
        rf"\textbf{{System prompt.}} \texttt{{\small {esc(system)}}}\par" "\n"
    )

tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{fontspec}
\setmainfont{Arial}
\setsansfont{Arial}
\setmonofont{Consolas}[Scale=0.88]
\usepackage{amsmath,amssymb,booktabs,longtable,array,xcolor,microtype,hyperref,fancyhdr,titlesec,pdflscape}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}
\definecolor{ruleblue}{HTML}{0072B2}
\pagestyle{fancy}\fancyhf{}\lhead{CellSwarm-AD supplementary materials}\rhead{\thepage}
\setlength{\headheight}{14pt}
\titleformat{\section}{\large\bfseries\color{ruleblue}}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\sloppy
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\begin{document}
\begin{center}
{\LARGE\bfseries CellSwarm-AD Supplementary Materials}\par
\vspace{3pt}{\small IJMS revision -- code-derived and evidence-bounded supplement}
\end{center}
\vspace{5pt}\hrule\vspace{7pt}

All numerical entries are generated from executable repository configuration or are explicitly labelled as heuristic/model-assigned assumptions. No live large-language-model output is presented as experimental evidence.

\clearpage
\begin{landscape}
\section*{Supplementary Table S4. Code-derived parameter provenance}
\scriptsize
\begin{longtable}{P{0.36\textwidth} P{0.08\textwidth} P{0.24\textwidth} P{0.24\textwidth}}
\toprule
Parameter & Value & Unit & Provenance class \\
\midrule\endfirsthead
\toprule
Parameter & Value & Unit & Provenance class \\
\midrule\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\normalsize
The complete machine-readable table additionally records source detail, line number, code path, and declaration kind in \texttt{selected\_runtime\_parameters.csv}.
\end{landscape}

\clearpage
\section*{Supplementary Table S6. Layer 3 prompts and evidence boundary}
The following strings reproduce the prompt templates present in executable source. Braced fields are runtime substitutions. The repository default and fallback backend is deterministic mock software; no archived request/response log demonstrates that a live model generated the quantitative results in the manuscript.

""" + "\n".join(prompt_tex) + r"""

\textbf{Backend boundary.} The Anthropic adapter supplies a generic scientific-assistant system prompt only when its system-prompt argument is empty. The OpenAI adapter omits a system message when it is empty. The local backend posts only the prompt and maximum-token value. These adapters are software capabilities; they are not evidence that a live provider was used in the reported analyses.

\clearpage
\section*{Supplementary Table S7. Code-faithful Layer 0 agent updates}
This reconstruction follows the executable \texttt{step()} methods. Let $\Delta t$ denote the update interval, $A$ the amyloid-$\beta$ input, and
\[
\operatorname{clamp}_{[a,b]}(x)=\max\{a,\min(b,x)\}.
\]
Unless stated otherwise, states are code-normalized and do not have declared physical units.

\subsection*{Neuron agent}
With calcium $C$, tau phosphorylation $T$, and viability $V$, assignment order matters:
\begin{align}
C_{t+1}&=C_t+0.05A_t\Delta t,\\
T_{t+1}&=T_t+0.02C_{t+1}\Delta t,\\
V_{t+1}&=\max\{0,V_t-0.1T_{t+1}\Delta t\}.
\end{align}
There is no random term. Only viability has a lower clamp. Defaults are $C_0=0.1$, $T_0=0$, and $V_0=1$. The 12 \texttt{GENE\_MAP} values are model-assigned relative weights and are not transcriptomic measurements.

\subsection*{Astrocyte agent}
Let $R$ denote reactivity, $Y$ cytokine level, $G$ glutamate uptake, and $W$ calcium-wave output:
\begin{align}
R_{t+1}&=\min\{1,R_t+0.15(Y_t+0.1A_t)\Delta t\},\\
G_{t+1}&=\max\{0.1,0.8-0.5R_{t+1}\},\\
W_{t+1}&=0.3R_{t+1}.
\end{align}
There is no random term and no lower clamp on reactivity.

\subsection*{Microglia agent}
Let $N$ denote NF-$\kappa$B activity and $\epsilon_N\sim\mathcal N(0,(0.01\Delta t)^2)$:
\[
N_{t+1}=\operatorname{clamp}_{[0,1]}\left[N_t+(0.1A_t-0.05N_t)\Delta t+\epsilon_N\right].
\]
An independent $r\sim U(0,1)$ redraws activation state according to Table S7a.
\begin{center}
\textbf{Table S7a. Microglial state probabilities}\par\smallskip
\small
\begin{tabular}{lccc}\toprule
NF-$\kappa$B interval & $P(M1)$ & $P(M2)$ & $P(M0)$ \\\midrule
$N_{t+1}>0.6$ & 0.75 & 0.20 & 0.05 \\
$0.3<N_{t+1}\leq0.6$ & 0.30 & 0.40 & 0.30 \\
$0.1<N_{t+1}\leq0.3$ & 0.10 & 0.20 & 0.70 \\
$N_{t+1}\leq0.1$ & 0 & 0.05 & 0.95 \\\bottomrule
\end{tabular}
\end{center}
State-dependent raw phagocytosis is $0.1+\epsilon_1$ for M1, $0.6+\epsilon_2$ for M2, and $0.3+\epsilon_0$ for M0, followed by clamping to $[0,1]$. M1 outputs TNF-$\alpha=0.8N$ and IL-1$\beta=0.6N$; M2 outputs IL-10$=0.3N$ and TGF-$\beta=0.2N$.

\subsection*{Oligodendrocyte agent}
Let $M$ denote myelination capacity, $I$ myelin integrity, $P$ maturation progress, $C$ calcium, and $F$ TNF-$\alpha$:
\begin{align}
M_{t+1}&=\operatorname{clamp}_{[0,1]}(M_t-0.05A_t\Delta t-\epsilon_M),\\
I_{t+1}&=\operatorname{clamp}_{[0,1]}(I_t-0.08C_t\Delta t-\epsilon_I),\\
s_t&=1-\min(1,0.7F_t),\\
P_{t+1}&=\operatorname{clamp}_{[0,1]}[P_t+0.02s_t\Delta t+\epsilon_P].
\end{align}
The unidirectional transition is OPC to pre-OL at $P\geq0.4$, then pre-OL to mature-OL at $P\geq0.8$ on a later call.

\subsection*{Endothelial agent}
Let $J$ denote tight-junction strength, $B$ BBB integrity, $F$ TNF-$\alpha$, $L$ IL-10, and $Q$ net amyloid-$\beta$ transport:
\begin{align}
J_{t+1}&=\operatorname{clamp}_{[0,1]}\{J_t+[0.01(1-J_t)-0.04A_t]\Delta t+\epsilon_J\},\\
B_{t+1}&=\operatorname{clamp}_{[0,1]}\{B_t+[0.05(J_{t+1}-B_t)+0.06L_t(1-B_t)-0.10F_t]\Delta t+\epsilon_B\},\\
Q_{t+1}&=\operatorname{clamp}_{[-1,1]}[0.6B_{t+1}-0.3(1-B_{t+1})+\epsilon_Q].
\end{align}
The transport update replaces $Q_t$ rather than forming an Euler increment. Positive values denote clearance and negative values denote influx.

\subsection*{Implementation caveats}
Reported scalar states are rounded to four decimals by \texttt{get\_state()}, while internal updates retain floating-point precision. Input validation differs by class. Coefficients remain heuristic/model-assigned unless a direct parameter-level source is encoded.

\section*{Supplementary analysis files}
The accompanying archive contains patient-level virtual-trial visits, GEE terms, independent Monte Carlo trial summaries, APOE4 interaction output, ablation results, grid-convergence results, and SHA-256 provenance records under \texttt{paper/ijms\_revision/}.

\end{document}
"""

TEX.write_text(tex, encoding="utf-8")
print(TEX)

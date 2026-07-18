# Supplementary Table S6. Prompts actually present in Layer 3 code

This table contains only prompt strings found in executable source. Braced fields are runtime substitutions. A mojibake delimiter in `ExperimentChain.refine_experiment()` is rendered as an em dash for UTF-8 readability; no semantic text was added.

| ID | Code path | User prompt template | System prompt |
|---|---|---|---|
| S6-P1 | `layer3_orchestrator/chains/reasoning_chain.py`, `HYPOTHESIS` | `Based on the following AD simulation context, generate a testable hypothesis.\nContext: {context}\nFormulate a specific, mechanistic hypothesis:` | `You are an Alzheimer's Disease research scientist.` |
| S6-P2 | same file, `ANALYSIS` | `Analyze the following AD simulation results.\nResults: {context}\nProvide a detailed scientific analysis:` | same as S6-P1 |
| S6-P3 | same file, `SYNTHESIS` | `Synthesize the following findings into a coherent narrative.\nFindings: {context}\nSynthesis:` | same as S6-P1 |
| S6-P4 | same file, `CRITIQUE` | `Critically evaluate the following hypothesis and evidence.\nContent: {context}\nCritique:` | same as S6-P1 |
| S6-P5 | `layer3_orchestrator/chains/experiment_chain.py`, `design_experiment()` | `Design a simulation experiment to test this hypothesis:\nHypothesis: {hypothesis}\nAvailable parameters: {params_str}\nSpecify: experiment name, parameter ranges, duration, and metrics to track.` | `You are an AD computational experiment designer.` |
| S6-P6 | same file, `refine_experiment()` | `Refine this experiment based on results:\nOriginal: {spec.name} — {spec.description[:200]}\nResults: {results}\nSuggest refined parameters and metrics.` | empty string (the call supplies none) |

The Anthropic wrapper supplies `You are a scientific research assistant.` only when its `system_prompt` argument is empty. The OpenAI wrapper omits a system message when it is empty. The local backend posts only `prompt` and `max_tokens`; it does not transmit the system prompt.

## Default execution and evidentiary status

`configs/default.yaml` sets `backend: mock` and `model_name: mock-model`. `LLMConfig` and `Orchestrator` independently default to the mock backend. `main.py` reads the default backend and therefore exercises deterministic keyword-based mock responses unless the configuration is changed.

Repository-wide inspection found software adapters for OpenAI, Anthropic, and a local HTTP endpoint, but no archived request/response log, provider run identifier, pinned real model output, or result-generation provenance demonstrating that a real LLM produced the reported simulations or figures. Existing audit documents also state that real OpenAI/Anthropic paths were not tested. Consequently, Layer 3 must be described as a **software capability/proof-of-concept**. It must not be claimed to have generated, selected, or analyzed the reported numerical results without additional external evidence.

The mock backend returns fixed strings when the lowercased prompt contains `hypothesis`, `experiment`, `analyze`, or `result`; otherwise it returns the first 80 prompt characters prefixed by `Mock response to:`. These are test/demo outputs, not external model inference.

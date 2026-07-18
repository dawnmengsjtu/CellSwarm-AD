# Supplement reconstruction audit note

Audit date: 2026-07-16. Scope: repository executable source, configuration, outputs, and text documentation. No `.docx`, reviewer-response file, or original `paper/submission_package/supplementary_tables_ijms.md` was modified.

## Methods

1. Parsed YAML configuration leaves and Python AST declarations into reproducible CSV inventories.
2. Manually traced assignment order, random draws, clamp bounds, and branch thresholds in all five Layer 0 `step()` methods.
3. Enumerated prompt literals from `ReasoningChain` and `ExperimentChain`; traced backend defaults and provider adapters through `LLMWrapper`, `Orchestrator`, and `main.py`.
4. Searched repository outputs and documentation for real-provider run evidence. Textual claims were not treated as execution evidence.
5. Classified exact weights conservatively when a parameter-level literature derivation was absent.

## Findings and limitations

- S7 is a discrete-time transcription of current code, including sequential use of newly updated states, Gaussian noise scales, all clamps, and the full microglial piecewise categorical draw.
- S6 contains six real templates: four reasoning templates and two experiment templates. The six templates in the original submission supplement are not code-faithful and were not copied.
- The repository defaults to mock. Real-provider adapters establish capability only. No auditable evidence links a real LLM call to reported result generation.
- The source tree itself contains pre-existing mojibake in comments/docstrings and one prompt delimiter. New deliverables normalize scientific symbols and contain no Unicode replacement character or known mojibake marker.
- Units are often absent from executable agent code. The CSV says “code-normalized / not explicitly declared” rather than inventing physical units.
- Literature strings in `ADParameterDB` are reported as encoded provenance claims, not independently verified citations.

## Reproducibility boundary

The CSV extractor is deterministic given the source tree. Agent trajectories are not fully reproducible from configuration seed alone because Layer 0 stochastic agents use Python's module-level `random` generator and do not seed it internally. Any reproducibility claim for those agents requires externally calling `random.seed(...)` and recording that seed and execution order.

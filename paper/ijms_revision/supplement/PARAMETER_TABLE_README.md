# Runtime parameter tables: scope and provenance

Two machine-readable UTF-8 CSV files accompany this note:

- `selected_runtime_parameters.csv`: parameters declared in the five Layer 0 agent implementations and the central pathology coupling/cascade modules. This is the compact, methods-facing subset.
- `complete_runtime_parameters.csv`: all configuration leaves plus declared class/dataclass defaults and public function/method defaults in Layers 0–3, together with the encoded AD parameter-database records. This is the audit-facing inventory.

“Selected” is therefore a reporting subset of “complete”; it does not mean statistically selected, optimized, fitted, or literature validated. The complete inventory records declarations rather than every anonymous arithmetic literal. Anonymous coefficients that materially define the five agent equations are stated explicitly in S7.

Columns are `selection`, qualified `parameter`, `value`, `unit`, `source_type`, `source_detail`, repository-relative `code_path`, one-based `line` (zero for YAML leaves), and `declaration_kind`.

Source types are deliberately conservative:

- **heuristic/model-assigned**: numerical model defaults or weights without direct parameter-level provenance encoded at that declaration. This includes the neuron gene-expression weights and exact coefficients that are only biologically motivated by nearby comments.
- **configuration default**: a YAML-selected software/model value. Presence in YAML does not by itself prove that every key is consumed by a given entry point.
- **software default**: orchestration, sampling, or control behavior rather than a biological measurement.
- **literature-claimed database entry**: the code stores a citation string next to the value. This audit reports the encoded claim but does not independently validate the paper, unit conversion, or numerical interpretation.

The tables can be regenerated with:

```text
python paper/ijms_revision/supplement/extract_runtime_parameters.py
```

The extractor intentionally excludes tests and examples from the complete declaration inventory so test fixtures and demonstration-only choices are not mislabeled as model parameters. Example-specific values remain traceable in their source files.

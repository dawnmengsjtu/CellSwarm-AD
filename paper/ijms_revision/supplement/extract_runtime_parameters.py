#!/usr/bin/env python3
"""Extract declared runtime parameters from CellSwarm-AD source code.

The extractor intentionally captures configuration leaves, dataclass/class
defaults, and public function defaults. It does not treat local intermediate
constants as independently tunable parameters.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SCAN_DIRS = ["layer0_cell", "layer1_tissue", "layer2_pathology", "layer3_orchestrator"]

SELECTED_MODULES = {
    "layer0_cell/agents/neuron.py",
    "layer0_cell/agents/astrocyte.py",
    "layer0_cell/agents/microglia.py",
    "layer0_cell/agents/oligodendrocyte.py",
    "layer0_cell/agents/endothelial.py",
    "layer2_pathology/coupling/abeta_calcium.py",
    "layer2_pathology/coupling/calcium_tau.py",
    "layer2_pathology/coupling/abeta_microglia.py",
    "layer2_pathology/coupling/tau_neuron.py",
    "layer2_pathology/coupling/microglia_neuron.py",
    "layer2_pathology/coupling/cascade.py",
}

UNITS = {
    "dt": "simulation-time unit",
    "temperature": "dimensionless",
    "max_tokens": "tokens",
    "width": "grid cells",
    "height": "grid cells",
    "steps": "steps",
    "duration": "steps",
    "seed": "integer seed",
    "population_size": "individuals",
    "generations": "generations",
    "mutation_rate": "fraction",
}


def literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def serial(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def source_type(path: str, name: str) -> str:
    if path.startswith("configs/"):
        return "configuration default"
    if path.startswith("layer3_orchestrator/"):
        return "software default"
    # Numeric coefficients in model code are not direct measurements merely
    # because a nearby comment cites biological motivation.
    return "heuristic/model-assigned"


def unit_for(name: str) -> str:
    low = name.lower()
    for key, unit in UNITS.items():
        if low == key or low.endswith("." + key) or low.endswith("_" + key):
            return unit
    if any(x in low for x in ("prob", "rate", "coeff", "threshold", "strength", "integrity", "viability", "activity", "reactivity", "capacity", "progress", "concentration", "calcium", "tau", "nfkb")):
        return "code-normalized / not explicitly declared"
    return "not explicitly declared"


def add(rows: list[dict[str, str]], path: str, line: int, scope: str, name: str, value: Any, kind: str) -> None:
    if value is None or isinstance(value, (dict, set)):
        return
    if isinstance(value, (str, int, float, bool, list, tuple)):
        rows.append({
            "selection": "selected" if path in SELECTED_MODULES else "complete-only",
            "parameter": f"{scope}.{name}" if scope else name,
            "value": serial(value),
            "unit": unit_for(name),
            "source_type": source_type(path, name),
            "source_detail": "Value declared in executable code; no direct literature provenance encoded for this declaration.",
            "code_path": path,
            "line": str(line),
            "declaration_kind": kind,
        })


def scan_python(path: Path, rows: list[dict[str, str]]) -> None:
    rel = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    # Module-level maps such as neuron GENE_MAP are executable model weights.
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            value = literal(node.value)
            if isinstance(value, dict):
                for key, child in value.items():
                    add(rows, rel, node.lineno, node.target.id, str(key), child, "module mapping entry")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value:
                add(rows, rel, item.lineno, node.name, item.target.id, literal(item.value), "class/dataclass default")
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                args = item.args.args
                defaults = item.args.defaults
                for arg, default in zip(args[-len(defaults):] if defaults else [], defaults):
                    if arg.arg not in {"self", "cls"}:
                        add(rows, rel, default.lineno, f"{node.name}.{item.name}", arg.arg, literal(default), "public method default")


def scan_parameter_database(rows: list[dict[str, str]]) -> None:
    """Add literature-database records, preserving that these are encoded claims."""
    path = ROOT / "data_platform" / "parameter_db" / "ad_parameters.py"
    rel = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Parameter":
            continue
        kw = {item.arg: literal(item.value) for item in node.keywords if item.arg}
        name = kw.get("name")
        if not isinstance(name, str):
            continue
        rows.append({
            "selection": "complete-only",
            "parameter": f"ADParameterDB.{name}",
            "value": serial(kw.get("value")),
            "unit": str(kw.get("unit", "not explicitly declared")),
            "source_type": "literature-claimed database entry",
            "source_detail": f"{kw.get('source', 'not encoded')}; confidence={kw.get('confidence', 'not encoded')}; range=[{kw.get('min_value', 'NA')}, {kw.get('max_value', 'NA')}]. Citation was not independently verified in this code audit.",
            "code_path": rel,
            "line": str(node.lineno),
            "declaration_kind": "Parameter database record",
        })


def flatten_yaml(value: Any, prefix: str, path: str, rows: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_yaml(child, f"{prefix}.{key}" if prefix else str(key), path, rows)
    else:
        name = prefix.rsplit(".", 1)[-1]
        add(rows, path, 0, "config", prefix, value, "YAML configuration leaf")


def main() -> None:
    rows: list[dict[str, str]] = []
    for dirname in SCAN_DIRS:
        for path in sorted((ROOT / dirname).rglob("*.py")):
            if "__pycache__" not in path.parts:
                scan_python(path, rows)
    scan_parameter_database(rows)
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        flatten_yaml(yaml.safe_load(path.read_text(encoding="utf-8-sig")), "", path.relative_to(ROOT).as_posix(), rows)

    # Exact duplicate declarations can arise where a dataclass default is also
    # exposed as a method default; retain declarations because paths/scopes differ.
    rows.sort(key=lambda r: (r["code_path"], int(r["line"]), r["parameter"]))
    fields = list(rows[0])
    with (OUT / "complete_runtime_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "selected_runtime_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if r["selection"] == "selected")
    print(f"complete={len(rows)} selected={sum(r['selection'] == 'selected' for r in rows)}")


if __name__ == "__main__":
    main()

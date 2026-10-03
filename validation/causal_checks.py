"""Compatibilidad entre contrato, schema y DAG."""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from causal_model.graph import satisfies_backdoor
from validation.results import BLOCK, GRAPH, PASS, CheckResult


def check_graph_contract(
    contract: Dict[str, Any], graph: nx.DiGraph, feature_schema: Dict[str, Any]
) -> CheckResult:
    c = contract["causal"]
    t, y = c["treatment"], c["outcome"]
    covariates = c["pre_treatment_covariates"]
    problems = []

    schema_cov = [col["name"] for col in feature_schema["columns"] if col.get("role") == "pre_treatment_covariate"]
    if set(schema_cov) != set(covariates):
        problems.append("las covariables del schema no coinciden con las del contrato")
    if set(graph.nodes) != {t, y, *covariates}:
        problems.append("los nodos del DAG no coinciden con el contrato")
    descendants = set(covariates) & nx.descendants(graph, t)
    if descendants:
        problems.append(f"covariables descendientes del tratamiento: {sorted(descendants)}")
    if not satisfies_backdoor(graph, t, y, covariates):
        problems.append("el conjunto de ajuste no cumple el criterio de puerta trasera")
    if c["estimand"] != "ATE":
        problems.append(f"estimando '{c['estimand']}' no soportado por el estimador")

    if problems:
        return CheckResult(GRAPH, BLOCK, "; ".join(problems))
    return CheckResult(
        GRAPH, PASS, f"ajuste por {len(covariates)} covariables cumple el criterio de puerta trasera"
    )

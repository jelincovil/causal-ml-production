"""DAG causal derivado del contrato: confusores pre-tratamiento -> T, Y; T -> Y."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import networkx as nx


def build_dag(contract: Dict[str, Any]) -> nx.DiGraph:
    causal = contract["causal"]
    t, y = causal["treatment"], causal["outcome"]
    covariates = list(causal["pre_treatment_covariates"])

    g = nx.DiGraph()
    g.add_nodes_from([t, y, *covariates])
    for x in covariates:
        g.add_edge(x, t)
        g.add_edge(x, y)
    g.add_edge(t, y)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("El grafo derivado del contrato contiene ciclos")
    return g


def satisfies_backdoor(
    g: nx.DiGraph, treatment: str, outcome: str, adjustment: Iterable[str]
) -> bool:
    """Criterio de puerta trasera (Pearl): Z sin descendientes de T y Z bloquea todo camino
    de puerta trasera, es decir T y Y quedan d-separados en el grafo sin las aristas que salen de T."""
    z = set(adjustment)
    if z & nx.descendants(g, treatment):
        return False
    g_under = g.copy()
    g_under.remove_edges_from(list(g.out_edges(treatment)))
    return bool(nx.is_d_separator(g_under, {treatment}, {outcome}, z))


def to_gml(g: nx.DiGraph) -> str:
    return "\n".join(nx.generate_gml(g))


def to_dot(contract: Dict[str, Any]) -> str:
    """DOT para la interfaz: las covariables se colapsan en un solo nodo."""
    causal = contract["causal"]
    t, y = causal["treatment"], causal["outcome"]
    covariates = causal["pre_treatment_covariates"]
    label = f"X: {len(covariates)} covariables pre-tratamiento\\n({covariates[0]} ... {covariates[-1]})"
    return (
        "digraph G {\n"
        "  rankdir=LR;\n"
        '  node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontname="Helvetica"];\n'
        f'  X [label="{label}"];\n'
        f'  T [label="{t}\\n(tratamiento)", fillcolor="#fde9d9"];\n'
        f'  Y [label="{y}\\n(outcome)", fillcolor="#e2f0d9"];\n'
        "  X -> T;\n  X -> Y;\n  T -> Y;\n}\n"
    )

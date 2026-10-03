"""Tipos comunes: estado de cada verificacion y su severidad."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

PASS, WARN, BLOCK = "PASS", "WARN", "BLOCK"
REVIEW, ROBUST, SKIPPED = "REVIEW", "ROBUST", "SKIPPED"

# Solo PASS/WARN/BLOCK deciden si se permite estimar; REVIEW/ROBUST/SKIPPED son informativos.
SEVERITY = {PASS: 0, ROBUST: 0, REVIEW: 0, SKIPPED: 0, WARN: 1, BLOCK: 2}

SCHEMA = "Schema"
TREATMENT = "Treatment coding"
VERSIONS = "Versions & integrity"
GRAPH = "Graph / contract"
MISSING = "Missingness"
DOMAIN = "Operational domain"
POSITIVITY = "Positivity"
DRIFT = "Distribution shift"
SUTVA = "SUTVA diagnostics"
REFUTATION = "Refutation (release)"
SENSITIVITY = "Unobserved confounding"

# Filas comprobables con los datos de entrada y los artefactos, frente a filas que solo dan evidencia indirecta
# o no verificable. Toda fila que produce validate_runtime_data debe estar en exactamente uno de los dos grupos.
OPERATIONAL_CHECKS = (SCHEMA, TREATMENT, VERSIONS, GRAPH, MISSING, DOMAIN, POSITIVITY, DRIFT)
INDIRECT_CHECKS = (SUTVA, REFUTATION, SENSITIVITY)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


def action_to_status(action: str) -> str:
    return {"block": BLOCK, "warn": WARN}[action]


def worst(statuses: Iterable[str]) -> str:
    result = PASS
    for s in statuses:
        if SEVERITY[s] > SEVERITY[result]:
            result = s
    return result if SEVERITY[result] > 0 else PASS

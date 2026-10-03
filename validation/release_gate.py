"""Gates PASS / WARN / BLOCK: en runtime (datos de entrada) y en release (evidencia de validez)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from causal_model.graph import build_dag
from contracts import load_contract, load_feature_schema, load_thresholds
from monitoring.monitoring_report import (
    build_monitoring_report,
    check_domain,
    check_drift,
    check_missingness,
)
from validation import consistency_checks, causal_checks, positivity_checks, schema_checks, sensitivity_checks, sutva_checks
from validation.results import (
    BLOCK,
    DOMAIN,
    DRIFT,
    MISSING,
    PASS,
    POSITIVITY,
    REFUTATION,
    SENSITIVITY,
    OPERATIONAL_CHECKS,
    SEVERITY,
    SKIPPED,
    SUTVA,
    WARN,
    CheckResult,
    worst,
)

DATA_DEPENDENT = (MISSING, DOMAIN, POSITIVITY, DRIFT, SUTVA)


@dataclass
class ValidationResult:
    checks: List[CheckResult]
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return worst(c.status for c in self.checks if SEVERITY[c.status] > 0)

    @property
    def operational_status(self) -> str:
        """Peor estado entre las verificaciones comprobables (excluye refutacion, sensibilidad y SUTVA)."""
        return worst(c.status for c in self.checks if c.name in OPERATIONAL_CHECKS)

    @property
    def has_blocking_error(self) -> bool:
        return self.status == BLOCK

    @property
    def effect_estimation(self) -> str:
        return "BLOCKED" if self.has_blocking_error else "ALLOWED"

    def get(self, name: str) -> Optional[CheckResult]:
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "effect_estimation": self.effect_estimation,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "metrics": c.metrics}
                for c in self.checks
            ],
        }


def _release_rows(release_report: Optional[Dict[str, Any]]) -> List[CheckResult]:
    if not release_report:
        return [
            CheckResult(REFUTATION, SKIPPED, "sin reporte de release"),
            CheckResult(SENSITIVITY, SKIPPED, "sin reporte de release"),
        ]
    refs = release_report["refutation"]["refutations"]
    failed = [k for k, v in refs.items() if v["status"] != "PASS"]
    refutation = CheckResult(
        REFUTATION,
        PASS if not failed else WARN,
        "placebo, causa comun aleatoria y subconjunto de datos superados"
        if not failed
        else f"refutaciones no superadas: {failed}",
        {k: v["status"] for k, v in refs.items()},
    )
    sens = release_report["sensitivity"]
    return [
        refutation,
        CheckResult(SENSITIVITY, sens["status"], sens["detail"], sens["metrics"]),
    ]


def validate_runtime_data(
    data: pd.DataFrame,
    contract: Optional[Dict[str, Any]] = None,
    reference_profile: Optional[Dict[str, Any]] = None,
    model_metadata: Optional[Dict[str, Any]] = None,
    *,
    thresholds: Optional[Dict[str, Any]] = None,
    feature_schema: Optional[Dict[str, Any]] = None,
    release_report: Optional[Dict[str, Any]] = None,
    graph: Optional[nx.DiGraph] = None,
    model_path: Optional[Path] = None,
) -> ValidationResult:
    contract = contract or load_contract()
    thresholds = thresholds or load_thresholds()
    feature_schema = feature_schema or load_feature_schema()
    graph = graph if graph is not None else build_dag(contract)

    checks: List[CheckResult] = []
    schema = schema_checks.check_schema(data, contract, feature_schema)
    treatment = schema_checks.check_treatment(data, contract)
    checks += [schema, treatment]
    checks.append(consistency_checks.check_versions(contract, model_metadata or {}, model_path))
    checks.append(causal_checks.check_graph_contract(contract, graph, feature_schema))

    extra: Dict[str, Any] = {}
    if BLOCK in (schema.status, treatment.status):
        checks += [CheckResult(name, SKIPPED, "omitida: el esquema o el tratamiento no son validos") for name in DATA_DEPENDENT]
    else:
        profile = build_monitoring_report(data, contract, reference_profile)
        positivity, e, t = positivity_checks.check_positivity(data, contract, thresholds)
        checks += [
            check_missingness(profile, contract, thresholds),
            check_domain(profile, contract, thresholds),
            positivity,
            check_drift(profile, contract, thresholds),
            sutva_checks.check_sutva(data, contract),
        ]
        extra = {"profile": profile, "propensity": e, "treatment": t}

    checks += _release_rows(release_report)
    return ValidationResult(checks=checks, extra=extra)


def evaluate_release(
    *,
    contract: Dict[str, Any],
    thresholds: Dict[str, Any],
    identification: Dict[str, Any],
    reference_validation: ValidationResult,
    refutation: Dict[str, Any],
    sensitivity: CheckResult,
    synthetic: Dict[str, Any],
    consistency_problems: List[str],
    model_card_errors: List[str],
) -> Tuple[bool, List[str]]:
    """Condiciones obligatorias del release. Devuelve (ok, motivos de fallo)."""
    failures: List[str] = list(consistency_problems)

    if not identification["identified"]:
        failures.append("el estimando no es identificable a partir del DAG")
    if reference_validation.has_blocking_error:
        blocked = [c.name for c in reference_validation.checks if c.status == BLOCK]
        failures.append(f"la poblacion de referencia incumple el contrato: {blocked}")

    for name, level in contract["validation"]["refutation"].items():
        key = "placebo_treatment" if name == "placebo" else name
        result = refutation["refutations"].get(key)
        if level == "required" and (result is None or result["status"] != "PASS"):
            failures.append(f"refutacion requerida no superada: {key}")

    if contract["validation"]["sensitivity"]["required"] and sensitivity.metrics.get("rv_q1_alpha05") is None:
        failures.append("falta el analisis de sensibilidad requerido")

    if not synthetic["passed"]:
        failures.append(
            f"prueba sintetica: error absoluto {synthetic['abs_error']:.3f} > {synthetic['max_abs_error']}"
        )
    failures += [f"model card: {e}" for e in model_card_errors]
    return (not failures, failures)

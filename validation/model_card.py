"""Model card causal machine-readable (causal-model-card-v1) generada desde la evidencia del release."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List

from contracts import MODEL_CARD_SCHEMA_PATH
from validation.assumptions import EMPIRICAL, INDIRECT, build_assumptions, card_item
from validation.release_gate import ValidationResult
from validation.results import CheckResult, REFUTATION


def _evidence(
    assumption, reference: ValidationResult, refutation: Dict[str, Any], sensitivity: CheckResult
) -> str:
    """Evidencia del release (poblacion de referencia). Solo niveles 1 y 2; el nivel 3 no tiene."""
    if assumption.status_row == REFUTATION:
        refs = refutation["refutations"]
        return "; ".join(f"{k}: {v['status']} (p={v['p_value']:.3f})" for k, v in refs.items())
    if assumption.id == "unobserved_confounding":
        return sensitivity.detail
    row = reference.get(assumption.status_row)
    return f"{row.detail} (poblacion de referencia, release)"


def build_supuestos(
    contract: Dict[str, Any],
    metadata: Dict[str, Any],
    reference: ValidationResult,
    refutation: Dict[str, Any],
    sensitivity: CheckResult,
) -> List[Dict[str, Any]]:
    items = []
    for a in build_assumptions(contract):
        item = card_item(a)
        if a.level in (EMPIRICAL, INDIRECT):
            item["evidencia"] = _evidence(a, reference, refutation, sensitivity)
        if a.id == "unobserved_confounding":
            item["covariables"] = metadata["covariates"]
        items.append(item)
    return items


def build_model_card(
    metadata: Dict[str, Any],
    refutation: Dict[str, Any],
    contract: Dict[str, Any],
    reference: ValidationResult,
    sensitivity: CheckResult,
) -> Dict[str, Any]:
    effect = metadata["effect"]
    spec = metadata["estimator"]
    placebo = refutation["refutations"]["placebo_treatment"]
    return {
        "schema_version": "causal-model-card-v1",
        "estimando": {"tipo": metadata["estimand"], "formula": "E[Y(1) - Y(0)]"},
        "diseno_identificacion": {
            "tipo": "observacional",
            "metodo": f"DML-PLR con cross-fitting (K={spec['cv_folds']}, {spec['model_y']} nuisances), {spec['method']}",
        },
        "supuestos": build_supuestos(contract, metadata, reference, refutation, sensitivity),
        "poblacion": {
            "variables_delimitadoras": {"scenario_id": "synthetic_inference_ihdp_medium", "split": "train"},
            "fuente": f"data/dataset.csv, n={metadata['n_train']} (benchmark sintetico IHDP-style)",
        },
        "metricas": {
            "efecto": effect["ate"],
            "ic95": [effect["ci_low"], effect["ci_high"]],
            "n": metadata["n_train"],
            "spec": {"learner": spec["model_y"], "K": spec["cv_folds"], "dml_variant": "DML2"},
        },
        "validacion": {"placebo": {"paso": placebo["status"] == "PASS", "fecha": date.today().isoformat()}},
        "no_usar_para": [
            "Decisiones clinicas o de politica publica reales",
            "Poblaciones distintas del escenario synthetic_inference_ihdp_medium sin revisar el reporte de monitoring",
            "Decisiones individuales sin considerar el intervalo de confianza (el efecto es un ATE)",
        ],
        "version": {
            "modelo": f"{metadata['model_name']} v{metadata['model_version']}",
            "hash_datos_entrenamiento": f"sha256:{metadata['training_data_sha256']}",
        },
    }


def validate_model_card(card: Dict[str, Any]) -> List[str]:
    from jsonschema import Draft202012Validator

    schema = json.loads(MODEL_CARD_SCHEMA_PATH.read_text(encoding="utf-8"))
    return [f"{'/'.join(map(str, e.path)) or '<raiz>'}: {e.message}" for e in Draft202012Validator(schema).iter_errors(card)]

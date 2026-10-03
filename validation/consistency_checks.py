"""Auditoria de artefactos: versiones, hashes y metadatos de contrato, DAG y modelo.

No evalua la consistencia causal del tratamiento (que 'tratado' signifique lo mismo en entrenamiento y entrada)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from causal_model.estimator import file_sha256, library_versions
from validation.results import BLOCK, PASS, VERSIONS, WARN, CheckResult, worst


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def check_versions(
    contract: Dict[str, Any],
    metadata: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> CheckResult:
    c = contract["causal"]
    expected = {
        "model_name": contract["model"]["name"],
        "model_version": contract["model"]["version"],
        "treatment_name": c["treatment"],
        "treatment_definition_version": contract["versions"]["treatment_definition"],
        "allowed_values": c["treatment_levels"],
        "preprocessing_version": contract["versions"]["preprocessing"],
        "causal_graph_version": contract["versions"]["causal_graph"],
        "covariates": c["pre_treatment_covariates"],
    }
    issues = [(BLOCK, f"{k}: contrato={v!r} vs modelo={metadata.get(k)!r}") for k, v in expected.items() if metadata.get(k) != v]

    if model_path is not None:
        if not model_path.is_file():
            issues.append((BLOCK, f"no existe el artefacto {model_path.name}"))
        elif file_sha256(model_path) != metadata.get("model_sha256"):
            issues.append((BLOCK, "el hash del artefacto no coincide con model_metadata.json"))

    installed = library_versions()
    for lib, trained in metadata.get("library_versions", {}).items():
        if lib == "python":
            continue
        if _major_minor(installed[lib]) != _major_minor(trained):
            issues.append((WARN, f"{lib}: entrenado con {trained}, instalado {installed[lib]}"))

    if not issues:
        return CheckResult(VERSIONS, PASS, "versiones, hashes y metadatos de contrato, DAG y modelo son compatibles (audita artefactos, no la consistencia causal del tratamiento)")
    return CheckResult(VERSIONS, worst(s for s, _ in issues), "; ".join(m for _, m in issues))

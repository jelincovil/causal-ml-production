"""Reporte de monitoring y su conversion en verificaciones PASS/WARN/BLOCK."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from monitoring.runtime_profile import build_runtime_profile
from validation.results import (
    BLOCK,
    DOMAIN,
    DRIFT,
    MISSING,
    PASS,
    WARN,
    CheckResult,
    action_to_status,
)


def build_monitoring_report(
    df: pd.DataFrame, contract: Dict[str, Any], reference: Dict[str, Any]
) -> Dict[str, Any]:
    return build_runtime_profile(df, contract, reference)


def check_missingness(profile: Dict[str, Any], contract: Dict[str, Any], thresholds: Dict[str, Any]) -> CheckResult:
    th, rate = thresholds["missingness"], profile["missing_max"]
    detail = f"maxima tasa de nulos en una covariable: {rate:.1%}"
    if rate >= th["critical_rate"]:
        return CheckResult(MISSING, action_to_status(contract["validation"]["missingness"]["action"]), detail)
    if rate >= th["warning_rate"]:
        return CheckResult(MISSING, WARN, detail + "; la estimacion usa solo casos completos")
    return CheckResult(MISSING, PASS, detail)


def check_domain(profile: Dict[str, Any], contract: Dict[str, Any], thresholds: Dict[str, Any]) -> CheckResult:
    th, rate = thresholds["domain"], profile["out_of_range_max"]
    detail = f"maxima fraccion de filas fuera de [q01, q99] de referencia en una variable: {rate:.1%}"
    if rate >= th["critical_out_of_range_rate"]:
        return CheckResult(DOMAIN, action_to_status(contract["validation"]["domain"]["action"]), detail)
    if rate >= th["warning_out_of_range_rate"]:
        return CheckResult(DOMAIN, WARN, detail)
    return CheckResult(DOMAIN, PASS, detail)


def check_drift(profile: Dict[str, Any], contract: Dict[str, Any], thresholds: Dict[str, Any]) -> CheckResult:
    th = thresholds["drift"]
    psi_max = profile["psi_max"]
    metrics = {
        "psi_max": psi_max,
        "psi_max_feature": profile["psi_max_feature"],
        "ks_max": profile["ks_max"],
        "treatment_rate": profile["treatment_rate"],
        "treatment_rate_reference": profile["treatment_rate_reference"],
    }
    detail = (
        f"PSI maximo {psi_max:.3f} ({profile['psi_max_feature']}); "
        f"tasa de tratamiento {profile['treatment_rate']:.3f} vs referencia {profile['treatment_rate_reference']:.3f}"
    )
    if psi_max >= th["critical_psi"]:
        status = action_to_status(contract["validation"]["drift"]["action"])
        return CheckResult(DRIFT, status, detail + " (nivel critical)", {**metrics, "level": "critical"})
    if psi_max >= th["warning_psi"]:
        return CheckResult(DRIFT, WARN, detail + " (nivel warning)", {**metrics, "level": "warning"})
    return CheckResult(DRIFT, PASS, detail, {**metrics, "level": "ok"})

"""Positividad: prueba ejecutable sobre P(T=1 | X)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from causal_model.estimator import complete_cases
from causal_model.propensity import crossfit_propensity, propensity_summary
from validation.results import BLOCK, PASS, POSITIVITY, WARN, CheckResult, action_to_status

MIN_ROWS = 50


def check_positivity(
    df: pd.DataFrame, contract: Dict[str, Any], thresholds: Dict[str, Any]
) -> Tuple[CheckResult, Optional[np.ndarray], Optional[np.ndarray]]:
    c = contract["causal"]
    th = thresholds["positivity"]
    d = complete_cases(df, contract)
    if len(d) < MIN_ROWS:
        return CheckResult(POSITIVITY, BLOCK, f"solo {len(d)} casos completos (minimo {MIN_ROWS})"), None, None

    X = d[c["pre_treatment_covariates"]].to_numpy(dtype=float)
    T = d[c["treatment"]].to_numpy(dtype=int)
    e = crossfit_propensity(X, T, seed=c["estimator"]["seed"])
    s = propensity_summary(e, T, warning_min=th["warning_min"], warning_max=th["warning_max"])

    summary = (
        f"propensity en [{s['min_propensity']:.3f}, {s['max_propensity']:.3f}]; "
        f"observaciones extremas {s['extreme_share']:.1%}"
    )
    if s["min_propensity"] < th["hard_min"] or s["max_propensity"] > th["hard_max"]:
        return CheckResult(POSITIVITY, BLOCK, f"{summary}; fuera de [{th['hard_min']}, {th['hard_max']}]", s), e, T
    band = contract["validation"]["positivity"]
    if s["min_propensity"] < band["min_propensity"] or s["max_propensity"] > band["max_propensity"]:
        status = action_to_status(band["action"])
        return CheckResult(POSITIVITY, status, f"{summary}; fuera de la banda [{band['min_propensity']}, {band['max_propensity']}]", s), e, T
    return CheckResult(POSITIVITY, PASS, summary, s), e, T

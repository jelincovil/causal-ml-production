"""Prueba causal sintetica: el estimador debe recuperar un efecto conocido bajo confusion observada."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from causal_model.estimator import ate_summary, fit_effect_model

TRUE_ATE = 2.0


def simulate(contract: Dict[str, Any], *, n: int = 1500, tau: float = TRUE_ATE, seed: int = 7) -> pd.DataFrame:
    """T depende de X (confusion observada) y Y = tau*T + g(X) + ruido: el ATE verdadero es tau."""
    c = contract["causal"]
    rng = np.random.default_rng(seed)
    covariates = c["pre_treatment_covariates"]
    X = rng.normal(size=(n, len(covariates)))
    propensity = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2])))
    t = rng.binomial(1, propensity)
    y = tau * t + 1.0 * X[:, 0] + 0.5 * X[:, 1] - 0.5 * X[:, 3] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=covariates)
    df[c["treatment"]] = t
    df[c["outcome"]] = y
    return df


def run_synthetic_recovery(contract: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    df = simulate(contract)
    est, n = fit_effect_model(df, contract)
    summary = ate_summary(est)
    abs_error = abs(summary["ate"] - TRUE_ATE)
    max_error = thresholds["synthetic_recovery"]["max_abs_error"]
    return {
        "true_ate": TRUE_ATE,
        "estimated_ate": summary["ate"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "abs_error": abs_error,
        "max_abs_error": max_error,
        "n": n,
        "passed": bool(abs_error <= max_error),
    }

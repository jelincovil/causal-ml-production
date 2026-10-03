"""Sensibilidad a confusion no observada: valor de robustez de Cinelli y Hazlett (2020)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import stats

from validation.results import REVIEW, ROBUST, SENSITIVITY, CheckResult


def robustness_value(t_stat: float, dof: int, *, q: float = 1.0, alpha: float = 0.05) -> float:
    """Fuerza minima (R2 parcial con T y con Y) que un confusor no observado necesita para
    reducir la estimacion en una fraccion q (q=1: a cero) con significancia alpha."""
    f_q = q * abs(t_stat) / np.sqrt(dof)
    if alpha < 1:
        f_crit = abs(stats.t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
        f_q = f_q - f_crit
    if f_q <= 0:
        return 0.0
    return float(0.5 * (np.sqrt(f_q**4 + 4 * f_q**2) - f_q**2))


def check_sensitivity(estimate: Dict[str, Any], thresholds: Dict[str, Any]) -> CheckResult:
    """`estimate` es el bloque runtime (o de referencia) con t_stat y dof."""
    rv = robustness_value(estimate["t_stat"], estimate["dof"], q=1.0, alpha=1.0)
    rv_alpha = robustness_value(estimate["t_stat"], estimate["dof"], q=1.0, alpha=0.05)
    robust = rv_alpha >= thresholds["sensitivity"]["robust_rv"]
    detail = (
        f"un confusor no observado tendria que explicar {rv:.1%} de la varianza residual de T y de Y "
        f"para anular la estimacion ({rv_alpha:.1%} para que el IC incluya 0). "
        "No prueba ausencia de confusion no observada."
    )
    return CheckResult(
        SENSITIVITY,
        ROBUST if robust else REVIEW,
        detail,
        {"rv_q1": rv, "rv_q1_alpha05": rv_alpha},
    )

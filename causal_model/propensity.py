"""Propensity score P(T=1 | X) con cross-fitting y resumen de overlap."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def crossfit_propensity(
    X: np.ndarray, T: np.ndarray, *, n_splits: int = 5, seed: int = 42, C: float = 1.0
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = LogisticRegression(C=C, max_iter=2000)
    return cross_val_predict(model, X, T, cv=cv, method="predict_proba")[:, 1]


def _ess_ratio(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights**2) / len(weights))


def propensity_summary(
    e: np.ndarray, T: np.ndarray, *, warning_min: float, warning_max: float
) -> Dict[str, Any]:
    treated = T == 1
    w_treated = 1.0 / e[treated]
    w_control = 1.0 / (1.0 - e[~treated])
    extreme = (e < warning_min) | (e > warning_max)
    return {
        "min_propensity": float(e.min()),
        "max_propensity": float(e.max()),
        "q01": float(np.quantile(e, 0.01)),
        "q99": float(np.quantile(e, 0.99)),
        "extreme_share": float(extreme.mean()),
        "ess_ratio_treated": _ess_ratio(w_treated),
        "ess_ratio_control": _ess_ratio(w_control),
        "n": int(len(e)),
    }

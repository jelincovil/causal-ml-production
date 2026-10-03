"""Metricas de distribution shift contra el perfil de referencia."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from monitoring.reference_profile import KS_PROBS, bin_proportions

EPS = 1e-4


def psi(values: np.ndarray, feature_ref: Dict[str, Any]) -> float:
    expected = np.clip(np.asarray(feature_ref["psi_expected"]), EPS, None)
    actual = np.clip(bin_proportions(values, np.asarray(feature_ref["psi_edges"])), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_vs_reference(values: np.ndarray, feature_ref: Dict[str, Any]) -> float:
    """KS de una muestra contra la CDF de referencia, aproximada con su malla de cuantiles."""
    grid = np.asarray(feature_ref["ks_grid"])
    ecdf = np.searchsorted(np.sort(values), grid, side="right") / len(values)
    return float(np.max(np.abs(ecdf - KS_PROBS)))


def feature_drift_table(df: pd.DataFrame, reference: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, ref in reference["features"].items():
        values = df[name].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "feature": name,
                "psi": psi(values, ref),
                "ks": ks_vs_reference(values, ref),
                "mean_shift_sd": float((values.mean() - ref["mean"]) / ref["std"]) if ref["std"] > 0 else 0.0,
                "out_of_range_rate": float(np.mean((values < ref["q01"]) | (values > ref["q99"]))),
                "missing_rate": float(df[name].isna().mean()),
            }
        )
    return pd.DataFrame(rows)

"""Perfil de la poblacion de referencia: parte del release, versionado en el repositorio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from contracts import ROOT

PROFILE_PATH = ROOT / "artifacts" / "monitoring" / "reference_profile.json"
N_BINS = 10
KS_PROBS = np.linspace(0.0, 1.0, 101)


def bin_proportions(values: np.ndarray, interior_edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(interior_edges, values, side="right")
    counts = np.bincount(idx, minlength=len(interior_edges) + 1)
    return counts / counts.sum()


def build_reference_profile(
    df: pd.DataFrame, contract: Dict[str, Any], *, data_version: str, data_sha256: str
) -> Dict[str, Any]:
    c = contract["causal"]
    features: Dict[str, Any] = {}
    for name in c["pre_treatment_covariates"]:
        values = df[name].dropna().to_numpy(dtype=float)
        interior = np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1])
        features[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "q01": float(np.quantile(values, 0.01)),
            "q99": float(np.quantile(values, 0.99)),
            "missing_rate": float(df[name].isna().mean()),
            "psi_edges": interior.tolist(),
            "psi_expected": bin_proportions(values, interior).tolist(),
            "ks_grid": np.quantile(values, KS_PROBS).tolist(),
        }
    return {
        "training_data_version": data_version,
        "training_data_sha256": data_sha256,
        "model_version": contract["model"]["version"],
        "n": int(len(df)),
        "treatment_rate": float((df[c["treatment"]] == 1).mean()),
        "outcome": {"mean": float(df[c["outcome"]].mean()), "std": float(df[c["outcome"]].std(ddof=1))},
        "features": features,
    }


def save_reference_profile(profile: Dict[str, Any], path: Path = PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=1) + "\n", encoding="utf-8")


def load_reference_profile(path: Path = PROFILE_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

"""Estimacion del efecto con EconML (LinearDML = PLR con cross-fitting) y persistencia del modelo."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from contracts import ROOT

MODEL_DIR = ROOT / "artifacts" / "model"
MODEL_FILE = "causal_model.joblib"
METADATA_FILE = "model_metadata.json"


@dataclass
class ModelBundle:
    estimator: Any
    metadata: Dict[str, Any]


def complete_cases(df: pd.DataFrame, contract: Dict[str, Any]) -> pd.DataFrame:
    c = contract["causal"]
    cols = [*c["pre_treatment_covariates"], c["treatment"], c["outcome"]]
    return df[cols].dropna()


def new_estimator(contract: Dict[str, Any]):
    from econml.dml import LinearDML

    spec = contract["causal"]["estimator"]
    seed = spec["seed"]
    return LinearDML(
        model_y=LassoCV(max_iter=5000, random_state=seed),
        model_t=LassoCV(max_iter=5000, random_state=seed),
        discrete_treatment=False,
        cv=spec["cv_folds"],
        random_state=seed,
    )


def fit_effect_model(df: pd.DataFrame, contract: Dict[str, Any]) -> Tuple[Any, int]:
    """Ajusta el estimador sobre los casos completos. Devuelve (estimador, n usado)."""
    c = contract["causal"]
    d = complete_cases(df, contract)
    est = new_estimator(contract)
    est.fit(
        d[c["outcome"]].to_numpy(dtype=float),
        d[c["treatment"]].to_numpy(dtype=float),
        X=None,
        W=d[c["pre_treatment_covariates"]].to_numpy(dtype=float),
    )
    return est, len(d)


def ate_summary(est: Any, alpha: float = 0.05) -> Dict[str, float]:
    inference = est.ate_inference()
    lo, hi = inference.conf_int_mean(alpha=alpha)
    return {
        "ate": float(est.ate()),
        "se": float(np.ravel(inference.stderr_mean)[0]),
        "ci_low": float(np.ravel(lo)[0]),
        "ci_high": float(np.ravel(hi)[0]),
    }


def estimate_effect(
    bundle: ModelBundle, data: pd.DataFrame, contract: Dict[str, Any]
) -> Dict[str, Any]:
    """Estimacion de referencia (modelo desplegado) y re-estimacion sobre la poblacion de entrada.

    Debe invocarse solo despues de validate_runtime_data(): no valida nada por si misma.
    """
    n_cov = len(contract["causal"]["pre_treatment_covariates"])
    reference = ate_summary(bundle.estimator)
    reference["n"] = int(bundle.metadata["n_train"])

    est, n = fit_effect_model(data, contract)
    runtime = ate_summary(est)
    runtime["n"] = n
    runtime["t_stat"] = runtime["ate"] / runtime["se"] if runtime["se"] > 0 else float("inf")
    runtime["dof"] = max(n - n_cov - 2, 1)
    return {
        "reference": reference,
        "runtime": runtime,
        "n_dropped": int(len(data) - n),
        "model_version": bundle.metadata["model_version"],
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("econml", "scikit-learn", "numpy", "pandas"):
        out[name] = importlib_metadata.version(name)
    return out


def build_metadata(
    est: Any, contract: Dict[str, Any], *, n_train: int, data_sha256: str
) -> Dict[str, Any]:
    c = contract["causal"]
    return {
        "model_name": contract["model"]["name"],
        "model_version": contract["model"]["version"],
        "treatment_name": c["treatment"],
        "treatment_definition_version": contract["versions"]["treatment_definition"],
        "allowed_values": c["treatment_levels"],
        "preprocessing_version": contract["versions"]["preprocessing"],
        "causal_graph_version": contract["versions"]["causal_graph"],
        "estimand": c["estimand"],
        "estimator": c["estimator"],
        "covariates": c["pre_treatment_covariates"],
        "n_train": int(n_train),
        "training_data_sha256": data_sha256,
        "effect": ate_summary(est),
        "library_versions": library_versions(),
    }


def save_bundle(est: Any, metadata: Dict[str, Any], directory: Path = MODEL_DIR) -> Dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILE
    joblib.dump(est, model_path)
    metadata = {**metadata, "model_sha256": file_sha256(model_path)}
    (directory / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def load_bundle(directory: Path = MODEL_DIR) -> ModelBundle:
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    return ModelBundle(estimator=joblib.load(directory / MODEL_FILE), metadata=metadata)

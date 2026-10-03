"""Refutaciones con DoWhy. Se ejecutan en el release y se guardan como artefacto JSON."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from causal_model.graph import to_gml

REFUTERS = (
    ("placebo_treatment", "placebo_treatment_refuter", {"placebo_type": "permute"}),
    ("random_common_cause", "random_common_cause", {}),
    ("data_subset", "data_subset_refuter", {"subset_fraction": 0.8}),
)


def run_refutations(
    df: pd.DataFrame,
    contract: Dict[str, Any],
    graph: nx.DiGraph,
    thresholds: Dict[str, Any],
    *,
    num_simulations: Optional[int] = None,
) -> Dict[str, Any]:
    from dowhy import CausalModel

    logging.getLogger("dowhy").setLevel(logging.ERROR)
    c = contract["causal"]
    seed = c["estimator"]["seed"]
    n_sim = num_simulations or thresholds["refutation"]["num_simulations"]
    p_min = thresholds["refutation"]["p_value_min"]
    np.random.seed(seed)

    cols = [*c["pre_treatment_covariates"], c["treatment"], c["outcome"]]
    model = CausalModel(
        data=df[cols].dropna(),
        treatment=c["treatment"],
        outcome=c["outcome"],
        graph=to_gml(graph),
    )
    identified = model.identify_effect(proceed_when_unidentifiable=False)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.econml.dml.LinearDML",
        control_value=0,
        treatment_value=1,
        method_params={
            "init_params": {
                "model_y": LassoCV(max_iter=5000, random_state=seed),
                "model_t": LassoCV(max_iter=5000, random_state=seed),
                "discrete_treatment": False,
                "cv": c["estimator"]["cv_folds"],
                "random_state": seed,
            },
            "fit_params": {},
        },
    )

    results: Dict[str, Any] = {}
    for key, method, kwargs in REFUTERS:
        r = model.refute_estimate(
            identified, estimate, method_name=method, num_simulations=n_sim, **kwargs
        )
        p_value = float(r.refutation_result["p_value"])
        results[key] = {
            "status": "PASS" if p_value > p_min else "FAIL",
            "original_effect": float(r.estimated_effect),
            "refuted_effect": float(r.new_effect),
            "p_value": p_value,
        }
    return {
        "model_version": contract["model"]["version"],
        "num_simulations": int(n_sim),
        "p_value_min": p_min,
        "original_effect": float(estimate.value),
        "refutations": results,
    }

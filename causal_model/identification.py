"""Identificacion causal con DoWhy a partir del DAG del contrato (uso offline / release)."""

from __future__ import annotations

import logging
from typing import Any, Dict

import networkx as nx
import pandas as pd

from causal_model.graph import to_gml


def identify_with_dowhy(
    df: pd.DataFrame, contract: Dict[str, Any], graph: nx.DiGraph
) -> Dict[str, Any]:
    from dowhy import CausalModel

    logging.getLogger("dowhy").setLevel(logging.ERROR)
    c = contract["causal"]
    cols = [*c["pre_treatment_covariates"], c["treatment"], c["outcome"]]
    model = CausalModel(
        data=df[cols],
        treatment=c["treatment"],
        outcome=c["outcome"],
        graph=to_gml(graph),
    )
    identified = model.identify_effect(proceed_when_unidentifiable=False)
    backdoor = sorted(identified.get_backdoor_variables())
    return {
        "identified": bool(backdoor),
        "estimand_type": str(identified.estimand_type),
        "strategy": "backdoor",
        "backdoor_variables": backdoor,
    }

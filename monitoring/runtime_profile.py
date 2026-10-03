"""Perfil de la poblacion de entrada, comparable con el perfil de referencia."""

from __future__ import annotations

from typing import Any, Dict

from monitoring.drift import feature_drift_table


def build_runtime_profile(df, contract: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    t = contract["causal"]["treatment"]
    table = feature_drift_table(df, reference)
    worst_psi = table.loc[table["psi"].idxmax()]
    return {
        "n": int(len(df)),
        "treatment_rate": float((df[t] == 1).mean()),
        "treatment_rate_reference": reference["treatment_rate"],
        "psi_max": float(worst_psi["psi"]),
        "psi_max_feature": str(worst_psi["feature"]),
        "ks_max": float(table["ks"].max()),
        "out_of_range_max": float(table["out_of_range_rate"].max()),
        "missing_max": float(table["missing_rate"].max()),
        "table": table,
    }

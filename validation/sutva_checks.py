"""SUTVA: diagnosticos de senales de interferencia. No se afirma que SUTVA se verifique."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from validation.results import REVIEW, SUTVA, CheckResult

NOT_PROVABLE = "NOT PROVABLE"


def _icc_oneway(values: pd.Series, groups: pd.Series) -> float:
    g = values.groupby(groups)
    k, n = g.ngroups, len(values)
    if k < 2 or n <= k:
        return float("nan")
    sizes = g.size()
    n0 = (n - (sizes**2).sum() / n) / (k - 1)
    msb = (sizes * (g.mean() - values.mean()) ** 2).sum() / (k - 1)
    msw = ((values - g.transform("mean")) ** 2).sum() / (n - k)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else float("nan")


def check_sutva(df: pd.DataFrame, contract: Dict[str, Any]) -> CheckResult:
    cluster = contract["validation"]["sutva"].get("cluster_column")
    if not cluster or cluster not in df.columns:
        return CheckResult(
            SUTVA,
            REVIEW,
            f"{NOT_PROVABLE}: no hay variable de agrupacion (cluster_column) para buscar "
            "estructura de interferencia; la ausencia de interferencia requiere justificacion sustantiva",
            {"provable": False},
        )
    t = df[contract["causal"]["treatment"]]
    icc = _icc_oneway(t.astype(float), df[cluster])
    saturation = t.groupby(df[cluster]).mean()
    return CheckResult(
        SUTVA,
        REVIEW,
        f"{NOT_PROVABLE}: {df[cluster].nunique()} clusters; ICC del tratamiento {icc:.3f}; "
        f"saturacion maxima por cluster {saturation.max():.2f}. Revisar si hay derrames entre unidades",
        {"provable": False, "icc_treatment": icc, "max_saturation": float(saturation.max())},
    )

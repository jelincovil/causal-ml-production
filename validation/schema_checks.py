"""Schema de entrada y codificacion del tratamiento."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from validation.results import (
    BLOCK,
    PASS,
    SCHEMA,
    TREATMENT,
    CheckResult,
    action_to_status,
    worst,
)


def check_schema(
    df: pd.DataFrame, contract: Dict[str, Any], feature_schema: Dict[str, Any]
) -> CheckResult:
    rules = contract["validation"]["schema"]
    columns = feature_schema["columns"]
    expected = [c["name"] for c in columns]
    required = [c["name"] for c in columns if c.get("required", True)]

    missing = [c for c in required if c not in df.columns]
    unknown = [c for c in df.columns if c not in expected]
    non_numeric = [
        c["name"]
        for c in columns
        if c["name"] in df.columns and not pd.api.types.is_numeric_dtype(df[c["name"]])
    ]
    null_violations = [
        c["name"]
        for c in columns
        if c["name"] in df.columns and not c.get("nullable", True) and df[c["name"]].isna().any()
    ]

    issues: List[tuple] = []
    if missing:
        issues.append((action_to_status(rules["missing_required_columns"]), f"faltan columnas requeridas: {missing}"))
    if unknown:
        issues.append((action_to_status(rules["unknown_columns"]), f"columnas desconocidas: {unknown}"))
    if non_numeric:
        issues.append((BLOCK, f"columnas no numericas: {non_numeric}"))
    if null_violations:
        issues.append((BLOCK, f"nulos en columnas que no los admiten: {null_violations}"))

    metrics = {"n_rows": int(len(df)), "missing": missing, "unknown": unknown}
    if not issues:
        return CheckResult(SCHEMA, PASS, "esquema compatible con feature_schema.json", metrics)
    return CheckResult(SCHEMA, worst(s for s, _ in issues), "; ".join(m for _, m in issues), metrics)


def check_treatment(df: pd.DataFrame, contract: Dict[str, Any]) -> CheckResult:
    t = contract["causal"]["treatment"]
    levels = set(contract["causal"]["treatment_levels"])
    if t not in df.columns:
        return CheckResult(TREATMENT, BLOCK, f"columna de tratamiento '{t}' ausente")

    values = set(df[t].dropna().unique().tolist())
    invalid = sorted(values - levels)
    if invalid:
        status = action_to_status(contract["validation"]["treatment"]["invalid_values"])
        return CheckResult(TREATMENT, status, f"valores no permitidos {invalid}; permitidos {sorted(levels)}")
    if values != levels:
        return CheckResult(
            TREATMENT, BLOCK, f"solo aparece {sorted(values)}: sin ambos niveles no hay contraste causal"
        )
    rate = float((df[t] == sorted(levels)[-1]).mean())
    return CheckResult(TREATMENT, PASS, f"valores {sorted(levels)}; tasa de tratamiento {rate:.3f}", {"treatment_rate": rate})

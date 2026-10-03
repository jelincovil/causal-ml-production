"""Origen de los datos de entrada: referencia, escenarios de demostracion o CSV propio."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

SEED = 11


def _shift(cols, sd):
    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            out[c] = out[c] + sd * df[c].std()
        return out

    return apply


def _scale(cols, k):
    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            out[c] = out[c] * k
        return out

    return apply


def _broken_overlap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(SEED)
    out["treatment"] = ((out["x0"] > 0).astype(int) ^ (rng.random(len(out)) < 0.005)).astype(int)
    return out


def _unknown_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_id"] = np.arange(len(out))
    return out


def _invalid_treatment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = np.random.default_rng(SEED).choice(len(out), size=len(out) // 20, replace=False)
    out.loc[out.index[idx], "treatment"] = 2
    return out


def _with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = np.random.default_rng(SEED).choice(len(out), size=len(out) // 10, replace=False)
    out.loc[out.index[idx], "x5"] = np.nan
    return out


SCENARIOS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "Referencia (poblacion de entrenamiento)": lambda df: df.copy(),
    "Drift moderado (x0-x2 desplazadas 0.4 sd)": _shift(["x0", "x1", "x2"], 0.4),
    "Drift severo (x0-x2 desplazadas 1.0 sd)": _shift(["x0", "x1", "x2"], 1.0),
    "Fuera de dominio (x0-x2 multiplicadas por 3)": _scale(["x0", "x1", "x2"], 3.0),
    "Solapamiento roto (tratamiento determinado por x0)": _broken_overlap,
    "Nulos (10% en x5)": _with_nulls,
    "Esquema roto (columna desconocida)": _unknown_column,
    "Tratamiento invalido (5% con valor 2)": _invalid_treatment,
}


def get_input_data(reference_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    st.sidebar.header("Datos de entrada")
    scenario = st.sidebar.selectbox("Escenario de demostracion", list(SCENARIOS), key="scenario")
    uploaded = st.sidebar.file_uploader("... o suba un CSV", type="csv", key="upload")
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded), f"CSV subido: {uploaded.name}"
        except Exception as exc:  # entrada del usuario: se informa y se detiene
            st.sidebar.error(f"No se pudo leer el CSV: {exc}")
            st.stop()
    return SCENARIOS[scenario](reference_df), scenario

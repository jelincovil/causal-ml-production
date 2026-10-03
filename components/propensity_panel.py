from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from validation.results import CheckResult


def render_propensity_panel(check: Optional[CheckResult], e: Optional[np.ndarray], t: Optional[np.ndarray]) -> None:
    st.subheader("Positividad / overlap")
    if e is None or check is None:
        st.info("Sin propensity: la verificacion de positividad no se pudo calcular.")
        return
    m = check.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min propensity", f"{m['min_propensity']:.3f}")
    c2.metric("Max propensity", f"{m['max_propensity']:.3f}")
    c3.metric("Observaciones extremas", f"{m['extreme_share']:.1%}")
    c4.metric("ESS/n (tratados / control)", f"{m['ess_ratio_treated']:.2f} / {m['ess_ratio_control']:.2f}")

    bins = np.linspace(0, 1, 21)
    centers = np.round((bins[:-1] + bins[1:]) / 2, 3)
    hist = pd.DataFrame(
        {
            "control": np.histogram(e[t == 0], bins)[0],
            "tratado": np.histogram(e[t == 1], bins)[0],
        },
        index=centers,
    )
    st.bar_chart(hist)
    st.caption(
        "El propensity mostrado viene de una regresion logistica con cross-fitting (5 folds), distinta del LassoCV que "
        "usa el estimador para E[T|X]. La positividad se juzga bajo esa especificacion: si la asignacion real no es "
        "logistica y lineal en las covariables, los propensities extremos pueden subestimarse. "
        "Los umbrales son criterios del proyecto (thresholds.yaml)."
    )

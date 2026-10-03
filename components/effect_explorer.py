from __future__ import annotations

from typing import Any, Dict

import altair as alt
import pandas as pd
import streamlit as st

from validation.results import CheckResult


def render_effect_explorer(effect: Dict[str, Any], sensitivity: CheckResult) -> None:
    st.subheader("Efecto estimado (ATE)")
    ref, run = effect["reference"], effect["runtime"]
    c1, c2 = st.columns(2)
    c1.metric(
        "ATE de referencia (modelo desplegado)",
        f"{ref['ate']:.3f}",
        help=f"IC95 [{ref['ci_low']:.3f}, {ref['ci_high']:.3f}], n={ref['n']}",
    )
    c2.metric(
        "ATE re-estimado sobre la entrada",
        f"{run['ate']:.3f}",
        delta=f"{run['ate'] - ref['ate']:+.3f} vs referencia",
        delta_color="off",
        help=f"IC95 [{run['ci_low']:.3f}, {run['ci_high']:.3f}], n={run['n']}",
    )

    st.warning(
        "El ATE re-estimado sobre la entrada supone que los supuestos de identificacion (ignorabilidad, positividad, "
        "SUTVA y el DAG declarado) tambien se cumplen en esa poblacion. Ningun check de esta aplicacion lo garantiza: "
        "dominio y drift solo detectan diferencias observables en las covariables."
    )

    df = pd.DataFrame(
        [
            {"fuente": "referencia", "ate": ref["ate"], "lo": ref["ci_low"], "hi": ref["ci_high"]},
            {"fuente": "entrada", "ate": run["ate"], "lo": run["ci_low"], "hi": run["ci_high"]},
        ]
    )
    base = alt.Chart(df).encode(y=alt.Y("fuente:N", title=None))
    st.altair_chart(
        base.mark_rule().encode(x=alt.X("lo:Q", title="ATE (IC 95%)"), x2="hi:Q")
        + base.mark_point(filled=True, size=90).encode(x="ate:Q"),
        use_container_width=True,
    )
    if effect["n_dropped"]:
        st.caption(f"Se excluyeron {effect['n_dropped']} filas con nulos (casos completos).")

    st.markdown(f"**Sensibilidad:** {sensitivity.detail}")
    st.caption(
        "El ATE es un efecto promedio bajo los supuestos del contrato; no es un efecto garantizado "
        "para una unidad individual."
    )

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_monitoring_panel(profile: Dict[str, Any], history) -> None:
    st.subheader("Monitoring contra la poblacion de referencia")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PSI maximo", f"{profile['psi_max']:.3f}", help=f"variable: {profile['psi_max_feature']}")
    c2.metric("KS maximo (aprox.)", f"{profile['ks_max']:.3f}")
    c3.metric(
        "Tasa de tratamiento",
        f"{profile['treatment_rate']:.3f}",
        delta=f"{profile['treatment_rate'] - profile['treatment_rate_reference']:+.3f} vs referencia",
        delta_color="off",
    )
    c4.metric("Filas", f"{profile['n']:,}")

    table = profile["table"].sort_values("psi", ascending=False)
    st.bar_chart(table.set_index("feature")["psi"])
    with st.expander("Detalle por variable"):
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown("**Historial de evaluaciones**")
    df = history.read()
    if df.empty:
        st.caption("Sin registros todavia.")
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)
    if not history.persistent:
        st.caption(
            "Historial solo de esta sesion: Community Cloud no garantiza el disco local. "
            "Para persistencia real configure [history].url en st.secrets (ver docs/04_mlops_validez.md)."
        )

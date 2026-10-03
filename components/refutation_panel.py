from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def render_refutation_panel(release_report: Optional[Dict[str, Any]]) -> None:
    st.subheader("Refutaciones (evidencia del release)")
    if not release_report:
        st.info("No hay artifacts/validation/refutation_report.json: ejecute scripts/validate_release.py.")
        return
    ref = release_report["refutation"]
    rows = [
        {
            "refutador": name,
            "estado": r["status"],
            "efecto original": round(r["original_effect"], 3),
            "efecto refutado": round(r["refuted_effect"], 3),
            "p-valor": round(r["p_value"], 3),
        }
        for name, r in ref["refutations"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Modelo v{ref['model_version']}; {ref['num_simulations']} simulaciones por refutador (DoWhy). "
        "Son resultados offline sobre la poblacion de entrenamiento: no se recalculan con la entrada actual."
    )

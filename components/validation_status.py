"""Estado de validez causal: se muestra siempre, antes de cualquier estimacion."""

from __future__ import annotations

import streamlit as st

from validation.release_gate import ValidationResult
from validation.results import REFUTATION, SENSITIVITY, SUTVA

COLORS = {"PASS": "green", "ROBUST": "green", "WARN": "orange", "BLOCK": "red", "REVIEW": "blue", "SKIPPED": "gray"}


def badge(status: str) -> str:
    return f":{COLORS.get(status, 'gray')}[**{status}**]"


def render_validation_status(validation: ValidationResult) -> None:
    st.subheader("Causal validity status")
    rows = "\n".join(f"| {c.name} | {badge(c.status)} | {c.detail} |" for c in validation.checks)
    st.markdown("| Verificacion | Estado | Detalle |\n|---|---|---|\n" + rows)
    st.caption(
        "Versions & integrity audita artefactos (versiones, hashes y metadatos del modelo); no evalua la consistencia "
        "causal del tratamiento. Los supuestos y su nivel de verificacion estan en la seccion siguiente."
    )

    if validation.effect_estimation == "BLOCKED":
        st.error("EFFECT ESTIMATION: BLOCKED")
    elif validation.status == "WARN":
        st.warning("EFFECT ESTIMATION: ALLOWED con advertencias")
    else:
        st.success("EFFECT ESTIMATION: ALLOWED")

    with st.expander("Lectura del estado causal (que se comprueba y que solo tiene evidencia indirecta)"):
        rows_by_name = {name: validation.get(name) for name in (SUTVA, REFUTATION, SENSITIVITY)}

        def indirect(name: str) -> str:
            check = rows_by_name[name]
            return badge(check.status) if check else "n/d"

        st.markdown(
            f"- **Operational checks** (se comprueban con los datos y los artefactos): {badge(validation.operational_status)}\n"
            "- **Evidencia indirecta o no verificable** (no entra en el estado operacional): "
            f"SUTVA {indirect(SUTVA)}, refutaciones {indirect(REFUTATION)}, sensibilidad {indirect(SENSITIVITY)}\n"
            "- El estado global de arriba combina ambos grupos; el detalle de cada supuesto esta en la seccion siguiente."
        )

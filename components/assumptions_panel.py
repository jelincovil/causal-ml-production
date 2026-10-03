"""Seccion unica de supuestos: renderiza la seccion `supuestos` de model_card.json, sin texto propio."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from components.validation_status import badge
from validation.assumptions import LEVELS, UNVERIFIABLE
from validation.release_gate import ValidationResult

NOT_VERIFIED = ":gray[**NO VERIFICADO**]"


def _cell(text: str) -> str:
    """Escapa `|` para que no rompa la tabla; el texto mostrado es identico al de la card."""
    return text.replace("|", "\\|").replace("\n", " ")


def _state(item: Dict[str, Any], validation: ValidationResult) -> str:
    """Estado vivo de la verificacion asociada; los supuestos no verificables nunca muestran PASS."""
    row = item.get("fila_validacion")
    if item["nivel"] == UNVERIFIABLE or not row:
        return NOT_VERIFIED
    check = validation.get(row)
    return badge(check.status) if check else "n/d"


def render_assumptions(card: Dict[str, Any], validation: ValidationResult) -> None:
    st.subheader("Supuestos del analisis")
    st.caption(
        "Cada supuesto aparece una sola vez, en el nivel en que la aplicacion puede tratarlo. El texto es la seccion "
        "`supuestos` de artifacts/model/model_card.json (la misma declaracion formal del release); el estado se evalua "
        "en esta ejecucion. Solo informa: no cambia ningun veredicto PASS, WARN o BLOCK."
    )
    items: List[Dict[str, Any]] = card["supuestos"]
    for level, title, intro in LEVELS:
        group = [s for s in items if s["nivel"] == level]
        st.markdown(f"**{title}**")
        st.caption(intro)
        rows = "\n".join(
            f"| {_cell(s['nombre'])} | {s['ambito']} | {_state(s, validation)} | `{_cell(s['funcion'])}` | "
            f"{_cell(s['descripcion'])} |"
            for s in group
        )
        st.markdown(
            "| Supuesto | Ambito | Estado | Funcion que lo evalua | Que se comprueba |\n|---|---|---|---|---|\n" + rows
        )

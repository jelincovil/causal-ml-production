"""Streamlit: estimacion de efecto causal condicionada a la validez operacional del contrato.

Flujo: INPUT -> VALIDATE -> DECIDE -> ESTIMATE -> REPORT.
"""

from __future__ import annotations

import json

import streamlit as st

from causal_model.estimator import MODEL_DIR, MODEL_FILE, estimate_effect, load_bundle
from causal_model.graph import build_dag
from components.assumptions_panel import render_assumptions
from components.causal_graph_view import render_causal_graph
from components.data_input import get_input_data
from components.effect_explorer import render_effect_explorer
from components.monitoring_panel import render_monitoring_panel
from components.propensity_panel import render_propensity_panel
from components.refutation_panel import render_refutation_panel
from components.validation_status import render_validation_status
from contracts import ROOT, load_contract, load_dataset, load_feature_schema, load_thresholds
from monitoring.history import history_store, make_record
from monitoring.reference_profile import load_reference_profile
from validation.release_gate import validate_runtime_data
from validation.results import POSITIVITY, WARN
from validation.sensitivity_checks import check_sensitivity

VALIDATION_DIR = ROOT / "artifacts" / "validation"


@st.cache_resource
def load_artifacts():
    """Recursos de solo lectura. El cache optimiza carga; no sustituye la validacion causal."""
    release_report = None
    refutation_path = VALIDATION_DIR / "refutation_report.json"
    validation_path = VALIDATION_DIR / "validation_report.json"
    if refutation_path.is_file() and validation_path.is_file():
        release_report = {
            "refutation": json.loads(refutation_path.read_text(encoding="utf-8")),
            "sensitivity": json.loads(validation_path.read_text(encoding="utf-8"))["sensitivity"],
        }
    contract = load_contract()
    return {
        "contract": contract,
        "thresholds": load_thresholds(),
        "schema": load_feature_schema(),
        "graph": build_dag(contract),
        "bundle": load_bundle(),
        "card": json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8")),
        "reference": load_reference_profile(),
        "reference_df": load_dataset(),
        "release_report": release_report,
    }


def main() -> None:
    st.set_page_config(page_title="Causal ML en produccion", layout="wide")
    st.title("Estimacion causal con validez operacional")
    st.caption(
        "Material docente con datos sinteticos (benchmark IHDP-style). "
        "No usar para decisiones clinicas ni de politica publica reales."
    )

    a = load_artifacts()
    data, source = get_input_data(a["reference_df"])
    st.markdown(f"**Entrada:** {source} ({len(data):,} filas)")

    validation = validate_runtime_data(
        data=data,
        contract=a["contract"],
        reference_profile=a["reference"],
        model_metadata=a["bundle"].metadata,
        thresholds=a["thresholds"],
        feature_schema=a["schema"],
        release_report=a["release_report"],
        graph=a["graph"],
        model_path=MODEL_DIR / MODEL_FILE,
    )
    render_validation_status(validation)
    render_assumptions(a["card"], validation)

    if validation.has_blocking_error:
        st.error("La estimacion causal ha sido bloqueada porque los datos no cumplen el contrato operacional.")
        st.stop()
    if validation.status == WARN:
        st.warning("La estimacion puede ejecutarse, pero existen condiciones que requieren revision.")

    render_causal_graph(a["contract"])
    render_propensity_panel(
        validation.get(POSITIVITY), validation.extra["propensity"], validation.extra["treatment"]
    )

    effect = estimate_effect(a["bundle"], data, a["contract"])
    sensitivity = check_sensitivity(effect["runtime"], a["thresholds"])

    history = history_store(st.secrets, st.session_state.setdefault("history", []))
    render_monitoring_panel(validation.extra["profile"], history)
    if st.button("Guardar esta evaluacion en el historial"):
        history.append(
            make_record(
                model_version=effect["model_version"],
                status=validation.status,
                profile=validation.extra["profile"],
                ate_runtime=effect["runtime"]["ate"],
            )
        )
        st.rerun()

    render_effect_explorer(effect, sensitivity)
    render_refutation_panel(a["release_report"])


main()

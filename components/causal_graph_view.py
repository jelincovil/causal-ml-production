from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from causal_model.graph import to_dot


def render_causal_graph(contract: Dict[str, Any]) -> None:
    st.subheader("DAG del contrato")
    st.graphviz_chart(to_dot(contract))
    st.caption(
        "Confusores pre-tratamiento afectan a T y a Y; T afecta a Y. "
        "El DAG es un supuesto declarado en contracts/causal_contract.yaml, no un resultado de los datos."
    )

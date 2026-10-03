import pytest
from streamlit.testing.v1 import AppTest

from components.data_input import SCENARIOS
from contracts import ROOT


def run_app(scenario=None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if scenario:
        at.sidebar.selectbox(key="scenario").select(scenario)
        at.run()
    return at


def labels(at):
    return [m.label for m in at.metric]


def test_reference_scenario_estimates_effect():
    at = run_app()
    assert not at.exception
    assert any("ATE re-estimado" in label for label in labels(at))
    assert not at.error


def test_blocking_scenarios_stop_before_estimation():
    for name in [n for n in SCENARIOS if "Esquema roto" in n or "Fuera de dominio" in n or "Solapamiento" in n]:
        at = run_app(name)
        assert not at.exception, name
        assert any("bloqueada" in e.value for e in at.error), name
        assert not any("ATE re-estimado" in label for label in labels(at)), name


def test_warning_scenario_estimates_with_warning():
    at = run_app([n for n in SCENARIOS if n.startswith("Drift moderado")][0])
    assert not at.exception
    assert any("advertencias" in w.value or "revision" in w.value for w in at.warning)
    assert any("ATE re-estimado" in label for label in labels(at))


def test_history_button_records_an_evaluation():
    at = run_app()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state["history"]) == 1
